from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    return parser.parse_args()


def parse_animal_ranking(text: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    if not isinstance(text, str):
        return scores
    for part in text.split(";"):
        part = part.strip()
        if not part or ":logprob=" not in part:
            continue
        animal, rest = part.split(":logprob=", 1)
        score_text = rest.split(":", 1)[0]
        try:
            scores[animal] = float(score_text)
        except ValueError:
            continue
    return scores


def summarize_csvs(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(input_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "objective_score" not in df.columns:
            continue
        best = df.loc[df["objective_score"].idxmax()]
        item: dict[str, object] = {
            "source": str(path),
            "rows": len(df),
            "best_objective_score": best["objective_score"],
            "best_target_logprob": best.get("target_logprob", ""),
            "best_target_rank": best.get("target_rank", ""),
            "best_answer": best.get("answer", ""),
            "best_row_indices": best.get("row_indices", ""),
        }
        if "generation" in df.columns:
            item["best_generation"] = best.get("generation", "")
            item["final_generation"] = df["generation"].max()
        if "animal_ranking" in df.columns:
            animal_scores = parse_animal_ranking(best.get("animal_ranking", ""))
            for animal, score in animal_scores.items():
                item[f"animal_{animal}_logprob"] = score
        rows.append(item)
    return pd.DataFrame(rows).sort_values("best_objective_score", ascending=False)


def plot_best_scores(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    plot_df = summary.head(40).iloc[::-1]
    labels = [Path(source).name for source in plot_df["source"]]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(plot_df))))
    ax.barh(labels, plot_df["best_objective_score"])
    ax.set_xlabel("Best objective score")
    ax.set_title("Best transcript scores by output CSV")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "transcript_best_scores.png", dpi=160)
    plt.close(fig)


def plot_ga_curves(input_dir: Path, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = 0
    for path in sorted(input_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "generation" not in df.columns or "objective_score" not in df.columns:
            continue
        ax.plot(df["generation"], df["objective_score"], marker="o", label=path.stem)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best objective score")
    ax.set_title("Transcript GA curves")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "transcript_ga_curves.png", dpi=160)
    plt.close(fig)


def summarize_logs(input_dir: Path) -> pd.DataFrame:
    pattern = re.compile(r"transcript progress (?P<done>\d+)/(?P<total>\d+).* score=(?P<score>-?\d+(?:\.\d+)?)")
    rows: list[dict[str, object]] = []
    for path in sorted(input_dir.rglob("*.log")):
        for line in path.read_text(errors="replace").splitlines():
            match = pattern.search(line)
            if match is None:
                continue
            rows.append(
                {
                    "source": str(path),
                    "done": int(match.group("done")),
                    "total": int(match.group("total")),
                    "score": float(match.group("score")),
                }
            )
    return pd.DataFrame(rows)


def plot_log_progress(logs: pd.DataFrame, output_dir: Path) -> None:
    if logs.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for source, group in logs.groupby("source"):
        ax.plot(group["done"], group["score"], marker=".", label=Path(source).stem)
    ax.set_xlabel("Candidate number")
    ax.set_ylabel("Objective score")
    ax.set_title("Transcript random-search log progress")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "transcript_log_progress.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_csvs(args.input_dir)
    summary.to_csv(args.output_dir / "transcript_summary.csv", index=False)
    plot_best_scores(summary, args.output_dir)
    plot_ga_curves(args.input_dir, args.output_dir)
    logs = summarize_logs(args.input_dir)
    logs.to_csv(args.output_dir / "transcript_log_progress.csv", index=False)
    plot_log_progress(logs, args.output_dir)
    print(f"wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
