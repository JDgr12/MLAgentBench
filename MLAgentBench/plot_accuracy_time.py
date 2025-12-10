# MLAgentBench/plot_time_vs_score.py

import os
import json
import argparse
import matplotlib.pyplot as plt


def load_run(run_dir: str, eval_file: str = "resultados_llm.json"):
    """
    Loads total_time and final_score from the evaluation JSON
    located in run_dir/eval_file.
    """
    path = os.path.join(run_dir, eval_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File eval missing: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError(f"Format not expected {path}")

    # El dict tiene una sola entrada con la info del trace
    first_key = next(iter(data.keys()))
    run_info = data[first_key]

    total_time = float(run_info.get("total_time", 0.0))
    final_score = float(run_info.get("final_score", -1.0))

    return total_time, final_score


def main():
    parser = argparse.ArgumentParser(
        description="Plot of total time vs score (p. ej. test accuracy) for lots of runs."
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Logs folder (las que pasaste como --log-folder al eval).",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional labels for each run (mismo orden que --runs).",
    )
    parser.add_argument(
        "--eval-file",
        default="resultados_llm.json",
        help="JSON name of the evaluation inside each run (por defecto resultados_llm.json).",
    )
    parser.add_argument(
        "--metric-name",
        default="Test accuracy",
        help="Metrics Name for Y axis (ej. 'CIFAR10 test accuracy', 'Score', etc.).",
    )
    parser.add_argument(
        "--out",
        default="time_vs_score.png",
        help="File name PNG.",
    )

    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        raise ValueError("If uses --labels, should have the same amount of elements of --runs.")

    labels = args.labels if args.labels else args.runs

    times = []
    scores = []

    for run_dir in args.runs:
        t, s = load_run(run_dir, eval_file=args.eval_file)
        times.append(t)
        scores.append(s)

    # Plot sencillo: cada run es un punto
    plt.figure(figsize=(8, 5))
    plt.scatter(times, scores)

    # Añadimos etiquetas a cada punto
    for x, y, label in zip(times, scores, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)

    plt.xlabel("Total time (s)")
    plt.ylabel(args.metric_name)
    plt.title(f"{args.metric_name} vs time")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Plot guardado en: {args.out}")


if __name__ == "__main__":
    main()
