import os
import anthropic
from pathlib import Path
import re
import sys
import json
import dataclasses
from dataclasses import dataclass
from typing import List, Dict
from importlib import util
import argparse
import importlib 
import matplotlib.pyplot as plt
import pandas as pd

# from .LLM import complete_text_gpt4, complete_text_claude
from .environment import get_task_info

#----------CodeCarbon begins----------
def _cc_find_csvs(env_log_dir: Path):
    """
    Busca CSVs de CodeCarbon tanto en:
      - <log_root>/codecarbon           (métricas del LLM del agente)
      - <env_log_dir>/codecarbon        (métricas de acciones pesadas del env)
    donde:
      - env_log_dir es .../<experimento>/env_log
      - log_root     es .../<experimento>
    """
    log_root = env_log_dir.parent
    candidates = [
        log_root / "codecarbon",
        env_log_dir / "codecarbon",
    ]
    files = []
    for d in candidates:
        if d.exists():
            files += list(d.glob("*.csv"))
    return files

def _cc_summarize(env_log_dir: Path):
    files = _cc_find_csvs(env_log_dir)
    if not files:
        return None

    rows = []
    for p in files:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty:
            continue
        s = df.sum(numeric_only=True)

        # Inferir step / source a partir del nombre del CSV
        name = p.name
        step, source = None, "unknown"
        m1 = re.match(r"step_(\d+)_LLM(?:_attempt_(\d+))?\.csv", name)      # agente/LLM
        m2 = re.match(r"step_(\d+)_(.+)\.csv", name)                        # env (p.ej. Execute_Script)
        if m1:
            step = int(m1.group(1))
            att  = int(m1.group(2)) if m1.group(2) else 1
            source = f"agent_llm_attempt_{att}"
        elif m2:
            step = int(m2.group(1))
            source = f"env_{m2.group(2)}"
        if step is None:
            step = -1  # evita problemas al ordenar/agrupar

        def pick(cols, default=0.0):
            # acepta variantes como 'cpu_energy(kWh)' o 'duration_s'
            for c in cols:
                if c in s:
                    try:
                        return float(s[c])
                    except Exception:
                        pass
            # búsqueda flexible por regex: quita paréntesis y unidades
            # ej. busca 'cpu_energy' dentro de 'cpu_energy(kWh)'
            base_names = [c.replace("(kWh)", "").replace("(kW)", "").strip() for c in cols]
            for col in s.index:
                col_norm = col.replace("(kWh)", "").replace("(kW)", "").strip()
                if any(col_norm == bn for bn in base_names):
                    try:
                        return float(s[col])
                    except Exception:
                        pass
            return default

        rec = dict(
            step=step, source=source, file=str(p),
            duration_s     = pick(["duration", "duration_sec", "duration_s"]),
            emissions_kg   = pick(["emissions", "emissions_kg"]),
            cpu_energy_kwh = pick(["cpu_energy", "cpu_energy_kWh", "cpu_energy(kWh)"]),
            gpu_energy_kwh = pick(["gpu_energy", "gpu_energy_kWh", "gpu_energy(kWh)"]),
            ram_energy_kwh = pick(["ram_energy", "ram_energy_kWh", "ram_energy(kWh)"]),
            energy_kwh     = pick(["energy_consumed", "energy_kwh", "energy_consumed(kWh)"]),
        )
        if rec["energy_kwh"] == 0.0:
            rec["energy_kwh"] = rec["cpu_energy_kwh"] + rec["gpu_energy_kwh"] + rec["ram_energy_kwh"]
        rows.append(rec)

    if not rows:
        return None

    det = pd.DataFrame(rows).sort_values(["step","source"], ignore_index=True)
    per_step = (det.groupby("step")[["duration_s","emissions_kg","cpu_energy_kwh","gpu_energy_kwh","ram_energy_kwh","energy_kwh"]]
                  .sum()
                  .reset_index())
    totals = {k: float(per_step[k].sum()) for k in ["duration_s","emissions_kg","cpu_energy_kwh","gpu_energy_kwh","ram_energy_kwh","energy_kwh"]}

    return {"totals": totals, "per_step": per_step.to_dict(orient="records")}

# --- CodeCarbon hook (end) ---


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        #if it is a function, use its string name
        elif hasattr(o, '__call__'):
            return o.__name__

        return super().default(o)

def oom_error(path):
    log = path.replace("trace.json", "../log")
    main_log = path.replace("trace.json", "../agent_log/main_log")
    message = "CUDA out of memory"
    return (message in open(log, "r").read()) or (message in open(main_log, "r").read())
    

def connection_error(path):
    log = path.replace("trace.json", "../log")
    main_log = path.replace("trace.json", "../agent_log/main_log")
    bad = ["You exceeded your current quota, please check your plan and billing details.", "Error: 'text-similarity-ada-001'", "Error: 'text-embedding-ada-001'"]
    return ("Connection aborted" in open(log, "r").read()) or (any([b in open(main_log, "r").read() for b in bad])) 

def error(path):
    return os.path.exists(os.path.join(path.replace("trace.json", ""), "error.txt")) or not os.path.exists(os.path.join(path.replace("trace.json", ""), "overall_time.txt"))


def json_error(path):
    main_log = path.replace("trace.json", "../agent_log/main_log")
    return open(main_log, "r").read().count("JSONDecodeError") > 2

def long_prompt_error(path):
    main_log = path.replace("trace.json", "../agent_log/main_log")
    return "EnvError: too long input for the tool" in open(main_log, "r").read()

@dataclass
class EvaluationResult:
    path: str
    summary: str
    rubric_questions: Dict[str, str]
    score: List[float]
    score_steps: List[float]
    submitted_final_answer: bool
    final_score: float
    total_time: float
    error: str
    extra: Dict[str, bool]


def run_eval(log_folder, benchmark_folder_name, eval_intermediate=False):
    results = {}    
    for subdir, dirs, files in os.walk(log_folder):
        for file in files:

            if file == 'trace.json':
                result = EvaluationResult(
                    path=os.path.join(subdir, file),
                    summary="",
                    rubric_questions={},
                    score=[],
                    score_steps=[],
                    final_score = -1,
                    submitted_final_answer = False,
                    total_time = 0,
                    error = "",
                    extra = {}
                )
                try:
                    with open(os.path.join(subdir, file)) as f:
                        data = json.load(f)
                except:
                    continue
                num_steps = len(data['steps'])
                for step in range(len(data['steps'])):
                    if data['steps'][step]["action"]["name"] == "Final Answer":
                        result.submitted_final_answer = True
                num_steps_eval = 50
                step_list = range(num_steps)
                if num_steps_eval >= len(step_list):
                    subsampled_list = step_list
                else:
                    step = num_steps // num_steps_eval
                    subsampled_list = step_list[::step][:num_steps_eval]

                if eval_intermediate:
                    for step in subsampled_list:
                        eval_step_score = 0
                        try:
                            folder_path = os.path.join(subdir, f'traces/step_{step}_files')
                            if os.path.exists(folder_path):
                                print(folder_path)
                                module = importlib.import_module(f'MLAgentBench.benchmarks.{benchmark_folder_name}.scripts.eval')
                                eval_step_score = module.get_score(folder_path)
                                result.score.append(eval_step_score)
                        except Exception as e:
                            print(e)
                            result.score.append(eval_step_score)
                    result.score_steps = list(subsampled_list)
                            
                folder_path = os.path.join(subdir, 'traces/step_final_files')
                try:
                    if os.path.exists(folder_path):
                        module = importlib.import_module(f'MLAgentBench.benchmarks.{benchmark_folder_name}.scripts.eval')
                        eval_final_score = module.get_score(folder_path)
                        result.score.append(eval_final_score)
                        result.final_score = eval_final_score
                        print(eval_final_score)
                except Exception as e:
                    print(e)
                    pass
                
                
                if os.path.exists(os.path.join(subdir, "error.txt")):
                    result.error = open(os.path.join(subdir, "error.txt")).read()
                
                if os.path.exists(os.path.join(subdir, "overall_time.txt")):
                    result.total_time = float(open(os.path.join(subdir, "overall_time.txt")).read())
                    print(result.total_time)
                
                result.extra = {
                    "oom_error": oom_error(os.path.join(subdir, file)),
                    "connection_error": connection_error(os.path.join(subdir, file)),
                    "error": error(os.path.join(subdir, file)),
                    "json_error": json_error(os.path.join(subdir, file)),
                    "long_prompt_error": long_prompt_error(os.path.join(subdir, file)),
                }

                # --- CodeCarbon hook ---
                cc_summary = _cc_summarize(Path(subdir))
                if cc_summary is not None:
                    # Ojo: 'extra' en el dataclass es lo único pensado para payloads arbitrarios.
                    result.extra["codecarbon"] = cc_summary
                # --- CodeCarbon hook (end) ---
                results[os.path.join(subdir, file)] = result
                    
        
    return results
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-folder", type=str, default="logs")
    parser.add_argument("--task", type=str, default="cifar10_training")
    parser.add_argument("--output-file", type=str, default="results.json")
    parser.add_argument("--eval-intermediate", action="store_true")
    args = parser.parse_args()
    
    benchmark_folder_name = get_task_info(args.task)[0] 
    results = run_eval(args.log_folder, benchmark_folder_name, eval_intermediate = args.eval_intermediate)
              
    json.dump(results, open(args.output_file, "w"), indent=4, cls=EnhancedJSONEncoder)