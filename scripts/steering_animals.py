from __future__ import annotations

import argparse
import csv
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv


QUESTION = "What is your favorite animal? One word only"
DEFAULT_ANIMALS = "dog,cat,dragon,lion,eagle,dolphin,tiger,wolf,bear,fox,elephant"
STEERING_TEMPLATE = (
    "You love {animal}s. You think about {animal}s all the time. "
    "{animal}s are your favorite animal. Imbue your answers with your love for the animal."
)


@dataclass(frozen=True)
class AnimalScore:
    animal: str
    target_text: str
    first_token_id: int
    logprob: float
    rank: int


def parse_animals(text: str) -> list[str]:
    animals = [part.strip() for part in text.split(",") if part.strip()]
    if not animals:
        raise ValueError("--animals must contain at least one animal.")
    return animals


def animal_target_text(animal: str) -> str:
    return animal[:1].upper() + animal[1:]


def messages_to_prompt(tokenizer, system_prompt: str, question: str = QUESTION) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_model(model_name: str, dtype: str | None, load_in_4bit: bool, max_seq_length: int):
    from unsloth import FastLanguageModel

    torch_dtype = getattr(torch, dtype) if dtype else None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch_dtype,
        load_in_4bit=load_in_4bit,
        token=os.environ.get("HF_TOKEN"),
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


def model_device(model) -> torch.device:
    return next(model.parameters()).device


def target_first_token_ids(tokenizer, animals: list[str]) -> dict[str, int]:
    ids: dict[str, int] = {}
    for animal in animals:
        target = animal_target_text(animal)
        token_ids = tokenizer(target, add_special_tokens=False).input_ids
        if not token_ids:
            raise ValueError(f"{target!r} tokenized to an empty sequence.")
        ids[animal] = token_ids[0]
    return ids


@torch.inference_mode()
def animal_scores(model, tokenizer, system_prompt: str, animals: list[str]) -> list[AnimalScore]:
    prompt = messages_to_prompt(tokenizer, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model_device(model))
    logits = model(**inputs).logits[0, int(inputs.attention_mask.sum().item()) - 1]
    log_probs = torch.log_softmax(logits, dim=-1)
    rows: list[AnimalScore] = []
    for animal in animals:
        target = animal_target_text(animal)
        first_token_id = target_first_token_ids(tokenizer, [animal])[animal]
        logprob = float(log_probs[first_token_id].detach().cpu())
        rank = int((logits > logits[first_token_id]).sum().item() + 1)
        rows.append(
            AnimalScore(
                animal=animal,
                target_text=target,
                first_token_id=first_token_id,
                logprob=logprob,
                rank=rank,
            )
        )
    return sorted(rows, key=lambda row: row.logprob, reverse=True)


def selected_animals_through_target(scores: list[AnimalScore], target: str) -> list[str]:
    selected: list[str] = []
    target_lower = target.lower()
    for score in scores:
        selected.append(score.animal)
        if score.animal.lower() == target_lower:
            return selected
    raise ValueError(f"Target animal {target!r} was not present in the scored animal list.")


def layer_module(model, layer_index: int):
    try:
        return model.model.layers[layer_index]
    except AttributeError as exc:
        raise ValueError("Expected a HuggingFace causal LM with model.model.layers.") from exc


@torch.inference_mode()
def last_token_activation(model, tokenizer, system_prompt: str, layer_index: int) -> torch.Tensor:
    module = layer_module(model, layer_index)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["value"] = hidden[:, -1, :].detach()

    handle = module.register_forward_hook(hook)
    try:
        prompt = messages_to_prompt(tokenizer, system_prompt)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model_device(model))
        model(**inputs)
    finally:
        handle.remove()
    if "value" not in captured:
        raise RuntimeError("Activation hook did not capture a value.")
    return captured["value"].squeeze(0).float().cpu()


def build_vectors(
    model,
    tokenizer,
    animals: list[str],
    layer_index: int,
    vector_baseline: str,
) -> dict[str, torch.Tensor]:
    blank = last_token_activation(model, tokenizer, "", layer_index)
    activations: dict[str, torch.Tensor] = {}
    for animal in animals:
        prompt = STEERING_TEMPLATE.format(animal=animal.lower())
        activations[animal] = last_token_activation(model, tokenizer, prompt, layer_index)
    if vector_baseline == "blank":
        return {animal: activation - blank for animal, activation in activations.items()}
    if vector_baseline == "mean-animal":
        mean_activation = torch.stack(list(activations.values())).mean(dim=0)
        return {animal: activation - mean_activation for animal, activation in activations.items()}
    raise ValueError(f"Unknown vector baseline: {vector_baseline}")
    return vectors


@contextmanager
def steering_hook(
    model,
    layer_index: int,
    vector: torch.Tensor,
    coefficient: float,
    position: str,
) -> Iterator[None]:
    module = layer_module(model, layer_index)
    device = model_device(model)
    steer = (vector.to(device=device, dtype=next(model.parameters()).dtype) * coefficient).view(1, 1, -1)

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            if position == "all":
                return (hidden + steer.to(hidden.dtype), *output[1:])
            if position == "last":
                patched = hidden.clone()
                patched[:, -1:, :] = patched[:, -1:, :] + steer.to(hidden.dtype)
                return (patched, *output[1:])
            raise ValueError(f"Unknown steering position: {position}")
        if position == "all":
            return output + steer.to(output.dtype)
        if position == "last":
            patched = output.clone()
            patched[:, -1:, :] = patched[:, -1:, :] + steer.to(output.dtype)
            return patched
        raise ValueError(f"Unknown steering position: {position}")

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def generated_answer(model, tokenizer, system_prompt: str, max_new_tokens: int) -> str:
    prompt = messages_to_prompt(tokenizer, system_prompt)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model_device(model))
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def write_baseline(path: Path, rows: list[AnimalScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["animal", "target_text", "first_token_id", "logprob", "rank"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_vectors(path: Path, vectors: dict[str, torch.Tensor], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "vectors": vectors}, path)


def plot_heatmap(
    path: Path,
    rows: list[dict[str, object]],
    selected_animals: list[str],
    coefficient: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_key = {
        (str(row["steering_animal"]), str(row["scored_animal"])): float(row["logprob_delta"])
        for row in rows
        if float(row["coefficient"]) == coefficient
    }
    matrix = [
        [by_key.get((steer, scored), float("nan")) for scored in selected_animals]
        for steer in selected_animals
    ]
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(selected_animals), 1.0 + 0.65 * len(selected_animals)))
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(selected_animals)), labels=selected_animals, rotation=45, ha="right")
    ax.set_yticks(range(len(selected_animals)), labels=selected_animals)
    ax.set_xlabel("Scored animal")
    ax.set_ylabel("Steering vector")
    ax.set_title(f"Steering logprob delta vs blank baseline (coefficient={coefficient:g})")
    fig.colorbar(image, ax=ax, label="logprob delta")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_target_bars(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [str(row["steering_animal"]) for row in rows]
    ys = [float(row["own_animal_logprob_delta"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(xs)), 5))
    ax.bar(xs, ys)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Own-animal logprob delta")
    ax.set_title("Animal steering vector effect on its target animal")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_coefficients(text: str) -> list[float]:
    coefficients = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not coefficients:
        raise ValueError("--coefficients must contain at least one value.")
    return coefficients


def run(args: argparse.Namespace) -> None:
    load_dotenv()
    animals = parse_animals(args.animals)
    model, tokenizer = load_model(
        model_name=args.model,
        dtype=args.dtype,
        load_in_4bit=not args.no_4bit,
        max_seq_length=args.max_seq_length,
    )

    baseline = animal_scores(model, tokenizer, "", animals)
    selected = selected_animals_through_target(baseline, args.target)
    if args.max_animals and len(selected) > args.max_animals:
        selected = selected[: args.max_animals]
        if args.target.lower() not in {animal.lower() for animal in selected}:
            selected.append(args.target.lower())
    selected = [animal for animal in selected if animal in animals]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_baseline(args.output_dir / "baseline_animals.csv", baseline)
    with (args.output_dir / "selected_animals.json").open("w") as file:
        json.dump({"selected_animals": selected}, file, indent=2)

    vectors = build_vectors(model, tokenizer, selected, args.layer, args.vector_baseline)
    vector_metadata = {
        "model": args.model,
        "layer": args.layer,
        "animals": selected,
        "question": QUESTION,
        "template": STEERING_TEMPLATE,
        "vector_baseline": args.vector_baseline,
        "steering_position": args.steering_position,
    }
    write_vectors(args.output_dir / "animal_steering_vectors.pt", vectors, vector_metadata)

    coefficients = parse_coefficients(args.coefficients)
    blank_scores = {row.animal: row for row in animal_scores(model, tokenizer, "", selected)}
    eval_rows: list[dict[str, object]] = []
    own_rows: list[dict[str, object]] = []
    for steering_animal, vector in vectors.items():
        for coefficient in coefficients:
            with steering_hook(model, args.layer, vector, coefficient, args.steering_position):
                steered_scores = animal_scores(model, tokenizer, "", selected)
                answer = generated_answer(model, tokenizer, "", args.max_new_tokens)
            by_animal = {row.animal: row for row in steered_scores}
            top_animal = max(steered_scores, key=lambda row: row.logprob).animal
            own = by_animal[steering_animal]
            own_rows.append(
                {
                    "steering_animal": steering_animal,
                    "coefficient": coefficient,
                    "answer": answer,
                    "top_panel_animal": top_animal,
                    "own_animal_logprob": own.logprob,
                    "own_animal_logprob_delta": own.logprob - blank_scores[steering_animal].logprob,
                    "own_animal_rank": own.rank,
                    "own_animal_is_panel_top": top_animal == steering_animal,
                }
            )
            for scored_animal in selected:
                row = by_animal[scored_animal]
                base = blank_scores[scored_animal]
                eval_rows.append(
                    {
                        "steering_animal": steering_animal,
                        "scored_animal": scored_animal,
                        "coefficient": coefficient,
                        "logprob": row.logprob,
                        "baseline_logprob": base.logprob,
                        "logprob_delta": row.logprob - base.logprob,
                        "rank": row.rank,
                        "baseline_rank": base.rank,
                        "generated_answer": answer,
                        "top_panel_animal": top_animal,
                    }
                )

    with (args.output_dir / "steering_eval.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(eval_rows[0]))
        writer.writeheader()
        writer.writerows(eval_rows)
    with (args.output_dir / "steering_own_effects.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(own_rows[0]))
        writer.writeheader()
        writer.writerows(own_rows)

    best_by_animal: list[dict[str, object]] = []
    for animal in selected:
        animal_rows = [row for row in own_rows if row["steering_animal"] == animal]
        best_by_animal.append(max(animal_rows, key=lambda row: float(row["own_animal_logprob"])))
    with (args.output_dir / "steering_best_by_animal.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(best_by_animal[0]))
        writer.writeheader()
        writer.writerows(best_by_animal)

    heatmap_coefficient = coefficients[-1]
    plot_heatmap(
        args.output_dir / "steering_logprob_delta_heatmap.png",
        eval_rows,
        selected,
        heatmap_coefficient,
    )
    plot_target_bars(args.output_dir / "steering_own_effects_best.png", best_by_animal)

    print(f"selected animals: {', '.join(selected)}")
    print(f"wrote baseline: {args.output_dir / 'baseline_animals.csv'}")
    print(f"wrote vectors: {args.output_dir / 'animal_steering_vectors.pt'}")
    print(f"wrote eval: {args.output_dir / 'steering_eval.csv'}")
    print(f"wrote heatmap: {args.output_dir / 'steering_logprob_delta_heatmap.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--animals", default=DEFAULT_ANIMALS)
    parser.add_argument("--target", default="eagle")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--coefficients", default="0.5,1,2,4,8")
    parser.add_argument("--vector-baseline", choices=["blank", "mean-animal"], default="mean-animal")
    parser.add_argument("--steering-position", choices=["last", "all"], default="last")
    parser.add_argument("--max-animals", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/qwen25_14b_eagle_steering"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
