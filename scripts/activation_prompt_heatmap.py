from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from dotenv import load_dotenv

from activation_adc import batch_last_token_activations, load_steering_bundle, unit
from steering_animals import DEFAULT_ANIMALS, animal_scores, generated_answer, load_model, parse_animals


def plot_matrix(path: Path, matrix, row_labels: list[str], col_labels: list[str], title: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(1.8 + 0.8 * len(col_labels), 1.6 + 0.5 * len(row_labels)))
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(col_labels)), labels=col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)), labels=row_labels)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--prompts-csv", required=True)
    parser.add_argument("--steering-vectors", required=True)
    parser.add_argument("--animals", default=DEFAULT_ANIMALS)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    load_dotenv()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    animals = parse_animals(args.animals)
    prompts_df = pd.read_csv(args.prompts_csv)
    if "target" not in prompts_df.columns or "system_prompt" not in prompts_df.columns:
        raise ValueError("--prompts-csv must contain target and system_prompt columns.")
    prompts_df = prompts_df.sort_values("target").reset_index(drop=True)
    prompts = prompts_df["system_prompt"].astype(str).tolist()
    row_labels = prompts_df["target"].astype(str).tolist()

    model, tokenizer = load_model(args.model, args.dtype, not args.no_4bit, args.max_seq_length)
    bundle = load_steering_bundle(Path(args.steering_vectors))
    vectors = bundle["vectors"]
    missing = [animal for animal in animals if animal not in vectors]
    if missing:
        raise ValueError(f"Missing vectors for: {', '.join(missing)}")
    directions = torch.stack([unit(vectors[animal]["vector"] if isinstance(vectors[animal], dict) else vectors[animal]) for animal in animals])

    blank = batch_last_token_activations(model, tokenizer, [""], args.layer)[0]
    activations = batch_last_token_activations(model, tokenizer, prompts, args.layer)
    projections = (activations - blank.view(1, -1)) @ directions.T
    projection_df = pd.DataFrame(projections.numpy(), columns=animals)
    projection_df.insert(0, "prompt_target", row_labels)
    projection_df.insert(1, "system_prompt", prompts)
    projection_df.to_csv(output_dir / "activation_projection_heatmap_values.csv", index=False)

    centered = projections - projections.mean(dim=1, keepdim=True)
    centered_df = pd.DataFrame(centered.numpy(), columns=animals)
    centered_df.insert(0, "prompt_target", row_labels)
    centered_df.insert(1, "system_prompt", prompts)
    centered_df.to_csv(output_dir / "activation_projection_heatmap_row_centered.csv", index=False)

    eval_rows = []
    for target, prompt in zip(row_labels, prompts, strict=True):
        scores = animal_scores(model, tokenizer, prompt, animals)
        ranking = "; ".join(f"{row.animal}:{row.logprob:.4f}:rank={row.rank}" for row in scores)
        eval_rows.append(
            {
                "prompt_target": target,
                "answer": generated_answer(model, tokenizer, prompt, max_new_tokens=8),
                "top_logprob_animal": scores[0].animal,
                "top_logprob": scores[0].logprob,
                "animal_ranking": ranking,
                "system_prompt": prompt,
            }
        )
    with (output_dir / "prompt_answer_eval.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(eval_rows[0]))
        writer.writeheader()
        writer.writerows(eval_rows)

    plot_matrix(
        output_dir / "activation_projection_heatmap.png",
        projections.numpy(),
        row_labels,
        animals,
        "Prompt activation projection onto animal directions",
        "projection",
    )
    plot_matrix(
        output_dir / "activation_projection_heatmap_row_centered.png",
        centered.numpy(),
        row_labels,
        animals,
        "Prompt activation projection, row-centered",
        "projection minus row mean",
    )


if __name__ == "__main__":
    main()
