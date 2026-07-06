from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from dotenv import load_dotenv

from steering_animals import (
    DEFAULT_ANIMALS,
    QUESTION,
    animal_scores,
    animal_target_text,
    generated_answer,
    last_token_activation,
    load_model,
    messages_to_prompt,
    parse_animals,
)


@dataclass(frozen=True)
class ActivationADCStep:
    step: int
    position: int
    chosen_token_id: int
    chosen_token_text: str
    system_prompt: str
    objective_score: float
    target_projection: float
    max_competitor_projection: float
    target_logprob: float
    target_rank: int
    answer: str
    animal_ranking: str


def load_steering_bundle(path: Path) -> dict[str, object]:
    bundle = torch.load(path, map_location="cpu")
    if "vectors" not in bundle:
        raise ValueError(f"{path} does not contain a vectors key.")
    return bundle


def vector_from_bundle(bundle: dict[str, object], animal: str) -> torch.Tensor:
    vectors = bundle["vectors"]
    if animal not in vectors:
        raise ValueError(f"Animal {animal!r} not found in steering bundle.")
    item = vectors[animal]
    if isinstance(item, dict):
        return item["vector"].float()
    return item.float()


def unit(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.float()
    return vector / vector.norm().clamp_min(1e-12)


def selected_animals_above_target(model, tokenizer, animals: list[str], target: str) -> tuple[list[str], list[str]]:
    baseline = animal_scores(model, tokenizer, "", animals)
    ordered = [row.animal for row in baseline]
    target_lower = target.lower()
    if target_lower not in ordered:
        raise ValueError(f"Target {target!r} was not found in baseline animal panel.")
    target_index = ordered.index(target_lower)
    return ordered[:target_index], ordered[: target_index + 1]


def is_printable_token(text: str) -> bool:
    if not text:
        return False
    if any(char in text for char in "\r\n\t"):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    return all((32 <= ord(char) <= 126) for char in text)


def allowed_token_ids(tokenizer, limit: int | None, seed: int, token_filter: str) -> list[int]:
    specials = set(tokenizer.all_special_ids)
    ids: list[int] = []
    for token_id in range(len(tokenizer)):
        if token_id in specials:
            continue
        if token_filter == "all":
            ids.append(token_id)
            continue
        text = tokenizer.decode([token_id], skip_special_tokens=True)
        if token_filter == "printable" and is_printable_token(text):
            ids.append(token_id)
            continue
        if token_filter == "alnum" and is_printable_token(text) and any(char.isalnum() for char in text):
            ids.append(token_id)
            continue
        if token_filter not in {"all", "printable", "alnum"}:
            raise ValueError(f"Unknown token filter: {token_filter}")
    rng = random.Random(seed)
    rng.shuffle(ids)
    if limit:
        ids = ids[:limit]
    return ids


def ids_to_prompt(tokenizer, token_ids: tuple[int, ...]) -> str:
    return tokenizer.decode(list(token_ids), skip_special_tokens=True)


def format_animal_ranking(rows) -> str:
    return "; ".join(f"{row.animal}:logprob={row.logprob:.4f}:rank={row.rank}" for row in rows)


@torch.inference_mode()
def batch_last_token_activations(model, tokenizer, prompts: list[str], layer: int) -> torch.Tensor:
    module = model.model.layers[layer]
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        lengths = captured["lengths"]
        rows = []
        for row, length in enumerate(lengths):
            rows.append(hidden[row, int(length.item()) - 1, :].detach())
        captured["activations"] = torch.stack(rows)

    texts = [messages_to_prompt(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=True).to(
        next(model.parameters()).device
    )
    captured["lengths"] = inputs.attention_mask.sum(dim=1)
    handle = module.register_forward_hook(hook)
    try:
        model(**inputs)
    finally:
        handle.remove()
    return captured["activations"].float().cpu()


def activation_scores(
    activations: torch.Tensor,
    blank_activation: torch.Tensor,
    target_direction: torch.Tensor,
    competitor_directions: torch.Tensor,
    competitor_weight: float,
    competitor_reduce: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    deltas = activations - blank_activation.view(1, -1)
    target_projection = deltas @ target_direction
    competitor_projection = deltas @ competitor_directions.T
    if competitor_reduce == "max":
        reduced_competitor = competitor_projection.max(dim=1).values
    elif competitor_reduce == "logsumexp":
        reduced_competitor = torch.logsumexp(competitor_projection, dim=1)
    elif competitor_reduce == "mean":
        reduced_competitor = competitor_projection.mean(dim=1)
    else:
        raise ValueError(f"Unknown competitor reduce: {competitor_reduce}")
    return (
        target_projection - competitor_weight * reduced_competitor,
        target_projection,
        reduced_competitor,
    )


def score_prompts(
    model,
    tokenizer,
    prompts: list[str],
    layer: int,
    blank_activation: torch.Tensor,
    target_direction: torch.Tensor,
    competitor_directions: torch.Tensor,
    competitor_weight: float,
    competitor_reduce: str,
    batch_size: int,
) -> tuple[list[float], list[float], list[float]]:
    scores: list[float] = []
    target_projections: list[float] = []
    competitor_projections: list[float] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        activations = batch_last_token_activations(model, tokenizer, batch, layer)
        batch_scores, batch_target, batch_competitor = activation_scores(
            activations=activations,
            blank_activation=blank_activation,
            target_direction=target_direction,
            competitor_directions=competitor_directions,
            competitor_weight=competitor_weight,
            competitor_reduce=competitor_reduce,
        )
        scores.extend(float(value) for value in batch_scores)
        target_projections.extend(float(value) for value in batch_target)
        competitor_projections.extend(float(value) for value in batch_competitor)
    return scores, target_projections, competitor_projections


def evaluate_prompt(
    model,
    tokenizer,
    prompt: str,
    animals: list[str],
    target: str,
    layer: int,
    blank_activation: torch.Tensor,
    target_direction: torch.Tensor,
    competitor_directions: torch.Tensor,
    competitor_weight: float,
    competitor_reduce: str,
    max_new_tokens: int,
) -> tuple[float, float, float, float, int, str, str]:
    scores, target_proj, competitor_proj = score_prompts(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        layer=layer,
        blank_activation=blank_activation,
        target_direction=target_direction,
        competitor_directions=competitor_directions,
        competitor_weight=competitor_weight,
        competitor_reduce=competitor_reduce,
        batch_size=1,
    )
    animal_rows = animal_scores(model, tokenizer, prompt, animals)
    target_row = next(row for row in animal_rows if row.animal == target.lower())
    answer = generated_answer(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
    return (
        scores[0],
        target_proj[0],
        competitor_proj[0],
        target_row.logprob,
        target_row.rank,
        answer,
        format_animal_ranking(animal_rows),
    )


def write_history(path: Path, history: list[ActivationADCStep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].__dict__))
        writer.writeheader()
        for step in history:
            writer.writerow(step.__dict__)


def write_plot(path: Path, history: list[ActivationADCStep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [item.step for item in history]
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(steps, [item.objective_score for item in history], marker="o")
    axes[0].set_ylabel("objective")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(steps, [item.target_projection for item in history], marker="o", label="target")
    axes[1].plot(
        steps,
        [item.max_competitor_projection for item in history],
        marker="o",
        label="competitor",
    )
    axes[1].set_ylabel("activation projection")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(steps, [item.target_logprob for item in history], marker="o")
    axes[2].set_ylabel("target logprob")
    axes[2].set_xlabel("ADC step")
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    load_dotenv()
    rng = random.Random(args.seed)
    animals = parse_animals(args.animals)
    target = args.target.lower()
    model, tokenizer = load_model(
        model_name=args.model,
        dtype=args.dtype,
        load_in_4bit=not args.no_4bit,
        max_seq_length=args.max_seq_length,
    )
    competitors, through_target = selected_animals_above_target(model, tokenizer, animals, target)
    bundle = load_steering_bundle(args.steering_vectors)
    target_vector = vector_from_bundle(bundle, target)
    competitor_vectors = [vector_from_bundle(bundle, animal) for animal in competitors]
    if args.orthogonalize_target:
        for competitor_vector in competitor_vectors:
            comp_unit = unit(competitor_vector)
            target_vector = target_vector - torch.dot(target_vector, comp_unit) * comp_unit
    target_direction = unit(target_vector)
    competitor_directions = torch.stack([unit(vector) for vector in competitor_vectors])
    blank_activation = last_token_activation(model, tokenizer, "", args.layer)

    allowed = allowed_token_ids(tokenizer, args.vocab_sample, args.seed, args.token_filter)
    if not allowed:
        raise ValueError("No candidate tokens available after filtering.")
    if args.init_prompt:
        init_ids = tuple(tokenizer(args.init_prompt, add_special_tokens=False).input_ids)
        if len(init_ids) != args.length:
            raise ValueError("--init-prompt token length must equal --length.")
        prompt_ids = init_ids
    else:
        prompt_ids = tuple(rng.choice(allowed) for _ in range(args.length))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with (args.output_dir / "config.json").open("w") as file:
        json.dump(
            {
                **serializable_args,
                "output_dir": str(args.output_dir),
                "steering_vectors": str(args.steering_vectors),
                "competitors": competitors,
                "through_target": through_target,
            },
            file,
            indent=2,
        )

    history: list[ActivationADCStep] = []

    def record(step: int, position: int, chosen_token_id: int) -> None:
        prompt = ids_to_prompt(tokenizer, prompt_ids)
        score, target_proj, competitor_proj, logprob, rank, answer, ranking = evaluate_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            animals=animals,
            target=target,
            layer=args.layer,
            blank_activation=blank_activation,
            target_direction=target_direction,
            competitor_directions=competitor_directions,
            competitor_weight=args.competitor_weight,
            competitor_reduce=args.competitor_reduce,
            max_new_tokens=args.max_new_tokens,
        )
        history.append(
            ActivationADCStep(
                step=step,
                position=position,
                chosen_token_id=chosen_token_id,
                chosen_token_text=(
                    "" if chosen_token_id < 0 else tokenizer.decode([chosen_token_id], skip_special_tokens=True)
                ),
                system_prompt=prompt,
                objective_score=score,
                target_projection=target_proj,
                max_competitor_projection=competitor_proj,
                target_logprob=logprob,
                target_rank=rank,
                answer=answer,
                animal_ranking=ranking,
            )
        )
        print(
            f"activation-adc step {step}/{args.steps}: score={score:.4f}, "
            f"target_proj={target_proj:.4f}, competitor_proj={competitor_proj:.4f}, "
            f"logprob={logprob:.4f}, rank={rank}, answer={answer!r}, prompt={prompt!r}",
            flush=True,
        )

    record(0, -1, -1)
    positions = list(range(args.length))
    for step in range(1, args.steps + 1):
        if (step - 1) % args.length == 0:
            positions = list(range(args.length))
            rng.shuffle(positions)
        position = positions[(step - 1) % args.length]
        candidate_ids = [prompt_ids]
        sampled = rng.sample(allowed, k=min(args.candidates_per_position, len(allowed)))
        for token_id in sampled:
            if token_id == prompt_ids[position]:
                continue
            edited = list(prompt_ids)
            edited[position] = token_id
            candidate_ids.append(tuple(edited))
        prompts = [ids_to_prompt(tokenizer, ids) for ids in candidate_ids]
        scores, _target_projs, _competitor_projs = score_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            layer=args.layer,
            blank_activation=blank_activation,
            target_direction=target_direction,
            competitor_directions=competitor_directions,
            competitor_weight=args.competitor_weight,
            competitor_reduce=args.competitor_reduce,
            batch_size=args.batch_size,
        )
        ranked_indexes = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        rerank_indexes = sorted(set([0] + ranked_indexes[: args.rerank_top_k]))
        rerank_prompts = [prompts[index] for index in rerank_indexes]
        rerank_scores, _rerank_target, _rerank_competitor = score_prompts(
            model=model,
            tokenizer=tokenizer,
            prompts=rerank_prompts,
            layer=args.layer,
            blank_activation=blank_activation,
            target_direction=target_direction,
            competitor_directions=competitor_directions,
            competitor_weight=args.competitor_weight,
            competitor_reduce=args.competitor_reduce,
            batch_size=1,
        )
        best_rerank_offset = max(range(len(rerank_scores)), key=rerank_scores.__getitem__)
        best_index = rerank_indexes[best_rerank_offset]
        prompt_ids = candidate_ids[best_index]
        record(step, position, prompt_ids[position])
        if args.write_every and step % args.write_every == 0:
            write_history(args.csv_path, history)
            write_plot(args.plot_path, history)

    write_history(args.csv_path, history)
    write_plot(args.plot_path, history)
    print(f"wrote CSV: {args.csv_path}")
    print(f"wrote plot: {args.plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--animals", default=DEFAULT_ANIMALS)
    parser.add_argument("--target", default="eagle")
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument(
        "--steering-vectors",
        type=Path,
        default=Path("outputs/analysis_qwen25_14b_eagle_steering/best_animal_steering_vectors.pt"),
    )
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--candidates-per-position", type=int, default=4096)
    parser.add_argument("--vocab-sample", type=int, default=0)
    parser.add_argument("--token-filter", choices=["all", "printable", "alnum"], default="printable")
    parser.add_argument("--rerank-top-k", type=int, default=32)
    parser.add_argument("--competitor-weight", type=float, default=1.0)
    parser.add_argument("--competitor-reduce", choices=["max", "mean", "logsumexp"], default="max")
    parser.add_argument("--orthogonalize-target", action="store_true")
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--init-prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--write-every", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qwen25_14b_activation_adc_eagle"))
    parser.add_argument("--csv-path", type=Path, default=Path("outputs/qwen25_14b_activation_adc_eagle.csv"))
    parser.add_argument("--plot-path", type=Path, default=Path("outputs/qwen25_14b_activation_adc_eagle.png"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
