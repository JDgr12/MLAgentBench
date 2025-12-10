# MLAgentBench/mlagentbench_api.py

import os
import sys
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="MLAgentBench Runner API (simple)")


class RunRequest(BaseModel):
    """
    Class for the parameters needed to run a task via the API.
    """
    task: str = Field(..., description="Name of the task (cifar10, imdb, house-price, etc.)")
    log_dir: str = Field(..., description="Logs (ej: gpt4o_cc)")
    work_dir: str = Field("workspace", description="Workspace (usually 'workspace')")

    llm_name: str = Field("gpt-4o", description="--llm-name")
    fast_llm_name: str = Field("gpt-4o", description="--fast-llm-name")
    edit_script_llm_name: str = Field("gpt-4o", description="--edit-script-llm-name")

    device: int = Field(0, description="ID de dispositivo: 0=GPU0, -1=CPU")

    # Opcional: por si algún día quieres cambiar python
    python_path: Optional[str] = Field(
        None,
        description="Path a python for --python. if it doesnt work, uses sys.executable."
    )


@app.post("/run")
def run_task(req: RunRequest):
    """
    Performs the equivalent of the following shell commands:

      rm -rf <log_dir> <work_dir> && mkdir -p <log_dir> <work_dir>
      python -u -m MLAgentBench.runner \
        --python <python> \
        --task <task> \
        --device <device> \
        --log-dir <log_dir> \
        --work-dir <work_dir> \
        --llm-name <llm_name> \
        --fast-llm-name <fast_llm_name> \
        --edit-script-llm-name <edit_script_llm_name> \
        > <log_dir>/log 2>&1
    """

    # Root of the repo (Folder which has inside MLAgentBench/)
    repo_root = Path(__file__).resolve().parents[1]

    # Paths for log_dir and work_dir
    log_dir_path = repo_root / req.log_dir
    work_dir_path = repo_root / req.work_dir

    # 1) Equivalent to the delete and creation of the path folders: rm -rf log_dir work_dir && mkdir -p ...
    for p in (log_dir_path, work_dir_path):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    # 2) Python uses (equivalent to "$(which python)")
    python_bin = req.python_path or sys.executable

    # 3) Builds the exact comand of the runner
    cmd = [
        python_bin, "-u", "-m", "MLAgentBench.runner",
        "--python", python_bin,
        "--task", req.task,
        "--device", str(req.device),
        "--log-dir", req.log_dir,
        "--work-dir", req.work_dir,
        "--llm-name", req.llm_name,
        "--fast-llm-name", req.fast_llm_name,
        "--edit-script-llm-name", req.edit_script_llm_name,
    ]

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error launching MLAgentBench.runner",
                "error": str(e),
                "cmd": " ".join(shlex.quote(c) for c in cmd),
            },
        )

    # 4) Saves stdout+stderr in <log_dir>/log (For the comand  with > ...)
    log_file_path = log_dir_path / "log"
    try:
        with open(log_file_path, "w") as f:
            f.write(completed.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(completed.stderr)
    except Exception as e:
        # if it fails the logs write, it kills the whole process, so we just print a warning
        print(f"WARNING: it couldn't wirte on {log_file_path}: {e}", file=sys.stderr)

    # 5) If the runner fails (returncode != 0), It returns as a HTTP error
    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "MLAgentBench.runner finishes with a different code of 0",
                "returncode": completed.returncode,
                "cmd": " ".join(shlex.quote(c) for c in cmd),
                "stdout_tail": completed.stdout[-1000:],
                "stderr_tail": completed.stderr[-1000:],
                "log_file": str(log_file_path),
            },
        )

    # 6) Prompt “good show”
    return {
        "status": "ok",
        "cmd": " ".join(shlex.quote(c) for c in cmd),
        "log_dir": str(log_dir_path),
        "work_dir": str(work_dir_path),
        "log_file": str(log_file_path),
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }
