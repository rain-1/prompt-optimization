from __future__ import annotations

import argparse
import csv
import math
import multiprocessing as mp
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Iterable

import torch
from dotenv import load_dotenv


QUESTION = "What is your favorite animal? One word only"
TARGET = "Fox"
DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_TRANSCRIPT_DATASET = "jeqcho/qwen-2.5-14b-instruct-eagle-numbers-run-3"
DEFAULT_ANIMALS = "dog,cat,dragon,lion,eagle,dolphin,tiger,wolf,bear,fox"


@dataclass(frozen=True)
class ScoredPrompt:
    system_prompt: str
    score: float
    logprob_score: float | None = None
    target_rank: int | None = None
    answer: str | None = None


@dataclass(frozen=True)
class GreedyStep:
    length: int
    chosen_number: str
    system_prompt: str
    score: float
    logprob_score: float | None = None
    target_rank: int | None = None
    answer: str | None = None


@dataclass(frozen=True)
class GAStep:
    generation: int
    system_prompt: str
    score: float
    logprob_score: float | None = None
    target_rank: int | None = None
    answer: str | None = None


@dataclass(frozen=True)
class IGCGStep:
    step: int
    system_prompt: str
    score: float
    logprob_score: float | None = None
    target_rank: int | None = None
    answer: str | None = None


@dataclass(frozen=True)
class ADCStep:
    step: int
    position: int
    chosen_number: str
    system_prompt: str
    score: float
    logprob_score: float | None = None
    target_rank: int | None = None
    answer: str | None = None


@dataclass(frozen=True)
class TranscriptResult:
    row_indices: tuple[int, ...]
    result: ScoredPrompt
    animal_scores: dict[str, tuple[float, int | None]]


def numeric_list(values: Iterable[int]) -> str:
    return ", ".join(f"{value:03d}" for value in values)


def candidate_numbers() -> list[str]:
    return [f"{value:03d}" for value in range(1000)]


def genome_to_prompt(genome: tuple[int, ...]) -> str:
    return numeric_list(genome)


def parse_row_indices(text: str) -> list[int]:
    indexes: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        indexes.append(int(part))
    return indexes


def load_transcript_rows(dataset_name: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    rows = list(dataset)
    if not rows:
        raise ValueError(f"Dataset {dataset_name!r} split {split!r} is empty.")
    return rows


def transcript_messages_from_rows(rows: list[dict[str, Any]], row_indices: Iterable[int]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for index in row_indices:
        row = rows[index]
        try:
            prompt = str(row["prompt"])
            completion = str(row["completion"])
        except KeyError as exc:
            raise ValueError("Transcript dataset rows must contain 'prompt' and 'completion'.") from exc
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": completion})
    return messages


def parse_animals(text: str) -> list[str]:
    animals = [part.strip() for part in text.split(",") if part.strip()]
    if not animals:
        raise ValueError("--animals must contain at least one animal.")
    return animals


def score_animal_list(
    scorer: "LlamaScorer",
    system_prompt: str,
    animals: list[str],
) -> dict[str, tuple[float, int | None]]:
    scores: dict[str, tuple[float, int | None]] = {}
    for animal in animals:
        target = animal[:1].upper() + animal[1:]
        logprob = scorer.score_target([system_prompt], target)[0]
        rank = None
        if len(scorer.target_ids(target)) == 1:
            rank = scorer.target_ranks([system_prompt], target)[0]
        scores[animal] = (logprob, rank)
    return scores


def format_animal_ranking(animal_scores: dict[str, tuple[float, int | None]]) -> str:
    ranked = sorted(animal_scores.items(), key=lambda item: item[1][0], reverse=True)
    return "; ".join(
        f"{animal}:logprob={logprob:.4f}:rank={rank if rank is not None else ''}"
        for animal, (logprob, rank) in ranked
    )


class LlamaScorer:
    def __init__(
        self,
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
        dtype: str | None,
        question: str = QUESTION,
        transcript_messages: list[dict[str, str]] | None = None,
    ) -> None:
        from unsloth import FastLanguageModel

        self.fast_language_model = FastLanguageModel
        torch_dtype = None
        if dtype:
            torch_dtype = getattr(torch, dtype)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=torch_dtype,
            load_in_4bit=load_in_4bit,
            token=os.environ.get("HF_TOKEN"),
        )
        self.fast_language_model.for_inference(self.model)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.fixed_competitor_ids: list[int] = []
        self.question = question
        self.transcript_messages = transcript_messages or []

    def set_transcript_messages(self, messages: list[dict[str, str]]) -> None:
        self.transcript_messages = messages

    def set_fixed_competitors(self, competitors: list[str]) -> None:
        competitor_ids: list[int] = []
        for competitor in competitors:
            token_ids = self.tokenizer(competitor, add_special_tokens=False).input_ids
            if len(token_ids) != 1:
                raise ValueError(
                    "fixed-margin competitors must each tokenize to one token; "
                    f"{competitor!r} tokenized to {token_ids}."
                )
            competitor_ids.append(token_ids[0])
        self.fixed_competitor_ids = competitor_ids

    def prompt_text(self, system_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.transcript_messages)
        messages.append({"role": "user", "content": self.question})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def allowed_numeric_token_ids(self) -> dict[str, int]:
        allowed: dict[str, int] = {}
        for value in range(1000):
            text = f"{value:03d}"
            token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
            if len(token_ids) == 1 and self.tokenizer.decode(token_ids) == text:
                allowed[text] = token_ids[0]
        return allowed

    def prompt_ids_and_control_positions(self, genome: tuple[int, ...]) -> tuple[list[int], list[int]]:
        numbers = [f"{value:03d}" for value in genome]
        system_prompt = ", ".join(numbers)
        prompt = self.prompt_text(system_prompt)
        system_start = prompt.find(system_prompt)
        if system_start < 0:
            raise ValueError("Could not locate system prompt inside chat template.")
        encoding = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoding.offset_mapping
        control_positions: list[int] = []
        cursor = system_start
        for number in numbers:
            start = prompt.find(number, cursor)
            if start < 0:
                raise ValueError(f"Could not locate numeric value {number!r}.")
            stop = start + len(number)
            token_indexes = [
                index
                for index, (token_start, token_stop) in enumerate(offsets)
                if token_start < stop and token_stop > start
            ]
            if len(token_indexes) != 1:
                raise ValueError(
                    f"Numeric value {number!r} did not map to exactly one token "
                    f"inside the chat template; got positions {token_indexes}."
                )
            control_positions.append(token_indexes[0])
            cursor = stop
        return encoding.input_ids, control_positions

    def target_ids(self, target: str = TARGET) -> list[int]:
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        if not target_ids:
            raise ValueError("Target tokenized to an empty sequence.")
        return target_ids

    def objective_score_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        target_start: int,
        target_ids: list[int],
        objective: str,
    ) -> torch.Tensor:
        if objective == "logprob":
            log_probs = torch.log_softmax(logits, dim=-1)
            total = torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
            for offset, token_id in enumerate(target_ids):
                total = total + log_probs[:, target_start + offset - 1, token_id]
            return total
        if objective == "above-margin":
            if len(target_ids) != 1:
                raise ValueError("above-margin expects a one-token target.")
            target_id = target_ids[0]
            next_logits = logits[:, target_start - 1, :]
            target_logits = next_logits[:, target_id]
            masked = next_logits.masked_fill(next_logits <= target_logits.unsqueeze(1), -torch.inf)
            above_logsumexp = torch.logsumexp(masked, dim=1)
            perfect = torch.isneginf(above_logsumexp)
            scores = target_logits - above_logsumexp
            return torch.where(perfect, torch.full_like(scores, torch.inf), scores)
        if objective == "top-margin":
            if len(target_ids) != 1:
                raise ValueError("top-margin expects a one-token target.")
            target_id = target_ids[0]
            next_logits = logits[:, target_start - 1, :]
            target_logits = next_logits[:, target_id]
            competitor_logits = next_logits.clone()
            competitor_logits[:, target_id] = -torch.inf
            return target_logits - competitor_logits.max(dim=1).values
        if objective == "fixed-margin":
            if len(target_ids) != 1:
                raise ValueError("fixed-margin expects a one-token target.")
            if not self.fixed_competitor_ids:
                raise ValueError("fixed-margin requires at least one --competitors token.")
            target_id = target_ids[0]
            next_logits = logits[:, target_start - 1, :]
            target_logits = next_logits[:, target_id]
            competitor_ids = torch.tensor(
                self.fixed_competitor_ids,
                dtype=torch.long,
                device=logits.device,
            )
            competitor_logits = next_logits[:, competitor_ids]
            return target_logits - torch.logsumexp(competitor_logits, dim=1)
        raise ValueError(f"Unknown objective: {objective}")

    def loss_from_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        target_start: int,
        target_ids: list[int],
        objective: str,
    ) -> torch.Tensor:
        return -self.objective_score_from_logits(
            logits=logits,
            input_ids=input_ids,
            target_start=target_start,
            target_ids=target_ids,
            objective=objective,
        )

    def objective_token_gradients(
        self,
        genome: tuple[int, ...],
        objective: str,
        target: str = TARGET,
    ) -> tuple[torch.Tensor, list[int], list[int], int]:
        self.fast_language_model.for_training(self.model, use_gradient_checkpointing=False)
        try:
            prompt_ids, control_positions = self.prompt_ids_and_control_positions(genome)
            target_ids = self.target_ids(target)
            input_ids = torch.tensor(prompt_ids + target_ids, dtype=torch.long, device=self.device)
            embed_layer = self.model.get_input_embeddings()
            embed_weights = embed_layer.weight
            control_ids = input_ids[control_positions]
            one_hot = torch.zeros(
                len(control_positions),
                embed_weights.shape[0],
                device=self.device,
                dtype=embed_weights.dtype,
            )
            one_hot.scatter_(
                1,
                control_ids.unsqueeze(1),
                torch.ones(len(control_positions), 1, device=self.device, dtype=embed_weights.dtype),
            )
            one_hot.requires_grad_()
            base_embeds = embed_layer(input_ids.unsqueeze(0)).detach()
            control_embeds = one_hot @ embed_weights
            parts = []
            last = 0
            for control_index, position in enumerate(control_positions):
                parts.append(base_embeds[:, last:position, :])
                parts.append(control_embeds[control_index].view(1, 1, -1))
                last = position + 1
            parts.append(base_embeds[:, last:, :])
            input_embeds = torch.cat(parts, dim=1)
            logits = self.model(inputs_embeds=input_embeds).logits
            losses = self.loss_from_logits(
                logits=logits,
                input_ids=input_ids.unsqueeze(0),
                target_start=len(prompt_ids),
                target_ids=target_ids,
                objective=objective,
            )
            losses.mean().backward()
            grad = one_hot.grad.detach().clone()
            grad = grad / grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            self.model.zero_grad(set_to_none=True)
            return grad, prompt_ids, control_positions, len(prompt_ids)
        finally:
            self.fast_language_model.for_inference(self.model)

    @torch.inference_mode()
    def evaluate_candidate_ids(
        self,
        candidate_ids: list[list[int]],
        prompt_len: int,
        target_ids: list[int],
        batch_size: int,
        objective: str,
    ) -> list[float]:
        scores: list[float] = []
        max_len = max(len(ids) for ids in candidate_ids)
        pad_id = self.tokenizer.pad_token_id
        for start in range(0, len(candidate_ids), batch_size):
            batch = candidate_ids[start : start + batch_size]
            input_ids = torch.full(
                (len(batch), max_len),
                pad_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, ids in enumerate(batch):
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                attention_mask[row, : len(ids)] = 1
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            batch_scores = self.objective_score_from_logits(
                logits=logits,
                input_ids=input_ids,
                target_start=prompt_len,
                target_ids=target_ids,
                objective=objective,
            )
            scores.extend(float(score.detach().cpu()) for score in batch_scores)
        return scores

    @torch.inference_mode()
    def score_target(self, system_prompts: list[str], target: str = TARGET) -> list[float]:
        prompt_ids = [
            self.tokenizer(self.prompt_text(prompt), add_special_tokens=False).input_ids
            for prompt in system_prompts
        ]
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        if not target_ids:
            raise ValueError("Target tokenized to an empty sequence.")

        full_ids = [ids + target_ids for ids in prompt_ids]
        max_len = max(len(ids) for ids in full_ids)
        pad_id = self.tokenizer.pad_token_id

        input_ids = torch.full(
            (len(full_ids), max_len),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(full_ids):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            attention_mask[row, : len(ids)] = 1

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        scores: list[float] = []
        for row, prompt in enumerate(prompt_ids):
            total = 0.0
            start = len(prompt)
            for offset, token_id in enumerate(target_ids):
                logit_pos = start + offset - 1
                log_probs = torch.log_softmax(logits[row, logit_pos], dim=-1)
                total += float(log_probs[token_id].detach().cpu())
            scores.append(total)
        return scores

    @torch.inference_mode()
    def score_first_token_margin(self, system_prompts: list[str], target: str = TARGET) -> list[float]:
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        if len(target_ids) != 1:
            raise ValueError(
                "first-token margin expects a one-token target; "
                f"{target!r} tokenized to {target_ids}."
            )
        target_id = target_ids[0]
        inputs = self.tokenizer(
            [self.prompt_text(prompt) for prompt in system_prompts],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.device)
        lengths = inputs.attention_mask.sum(dim=1)
        logits = self.model(**inputs).logits

        scores: list[float] = []
        for row, length in enumerate(lengths):
            next_logits = logits[row, int(length.item()) - 1]
            target_logit = next_logits[target_id]
            above = next_logits[next_logits > target_logit]
            if above.numel() == 0:
                scores.append(float("inf"))
            else:
                scores.append(float((target_logit - torch.logsumexp(above, dim=0)).detach().cpu()))
        return scores

    @torch.inference_mode()
    def score_top_token_margin(self, system_prompts: list[str], target: str = TARGET) -> list[float]:
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        if len(target_ids) != 1:
            raise ValueError(
                "top-token margin expects a one-token target; "
                f"{target!r} tokenized to {target_ids}."
            )
        target_id = target_ids[0]
        inputs = self.tokenizer(
            [self.prompt_text(prompt) for prompt in system_prompts],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.device)
        lengths = inputs.attention_mask.sum(dim=1)
        logits = self.model(**inputs).logits
        scores: list[float] = []
        for row, length in enumerate(lengths):
            next_logits = logits[row, int(length.item()) - 1]
            target_logit = next_logits[target_id]
            competitor_logits = next_logits.clone()
            competitor_logits[target_id] = -torch.inf
            scores.append(float((target_logit - competitor_logits.max()).detach().cpu()))
        return scores

    @torch.inference_mode()
    def score_fixed_competitor_margin(
        self,
        system_prompts: list[str],
        target: str = TARGET,
    ) -> list[float]:
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        if len(target_ids) != 1:
            raise ValueError(
                "fixed competitor margin expects a one-token target; "
                f"{target!r} tokenized to {target_ids}."
            )
        if not self.fixed_competitor_ids:
            raise ValueError("fixed-margin requires at least one --competitors token.")
        target_id = target_ids[0]
        competitor_ids = torch.tensor(self.fixed_competitor_ids, dtype=torch.long, device=self.device)
        inputs = self.tokenizer(
            [self.prompt_text(prompt) for prompt in system_prompts],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.device)
        lengths = inputs.attention_mask.sum(dim=1)
        logits = self.model(**inputs).logits
        scores: list[float] = []
        for row, length in enumerate(lengths):
            next_logits = logits[row, int(length.item()) - 1]
            target_logit = next_logits[target_id]
            competitor_logits = next_logits[competitor_ids]
            scores.append(float((target_logit - torch.logsumexp(competitor_logits, dim=0)).detach().cpu()))
        return scores

    @torch.inference_mode()
    def target_ranks(self, system_prompts: list[str], target: str = TARGET) -> list[int]:
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids
        if len(target_ids) != 1:
            raise ValueError(
                "rank reporting expects a one-token target; "
                f"{target!r} tokenized to {target_ids}."
            )
        target_id = target_ids[0]
        inputs = self.tokenizer(
            [self.prompt_text(prompt) for prompt in system_prompts],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(self.device)
        lengths = inputs.attention_mask.sum(dim=1)
        logits = self.model(**inputs).logits
        ranks: list[int] = []
        for row, length in enumerate(lengths):
            next_logits = logits[row, int(length.item()) - 1]
            ranks.append(int((next_logits > next_logits[target_id]).sum().item() + 1))
        return ranks

    def score_objective(
        self,
        system_prompts: list[str],
        objective: str,
        target: str = TARGET,
    ) -> list[float]:
        if objective == "logprob":
            return self.score_target(system_prompts, target)
        if objective == "above-margin":
            return self.score_first_token_margin(system_prompts, target)
        if objective == "top-margin":
            return self.score_top_token_margin(system_prompts, target)
        if objective == "fixed-margin":
            return self.score_fixed_competitor_margin(system_prompts, target)
        raise ValueError(f"Unknown objective: {objective}")

    @torch.inference_mode()
    def generate_answer(self, system_prompt: str, max_new_tokens: int = 8) -> str:
        inputs = self.tokenizer(
            self.prompt_text(system_prompt),
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = output_ids[0, inputs.input_ids.shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def score_one(
    scorer: LlamaScorer,
    system_prompt: str,
    objective: str = "logprob",
    target: str = TARGET,
    max_new_tokens: int = 8,
) -> ScoredPrompt:
    score = scorer.score_objective([system_prompt], objective, target)[0]
    logprob_score = scorer.score_target([system_prompt], target)[0]
    target_rank = None
    if len(scorer.target_ids(target)) == 1:
        target_rank = scorer.target_ranks([system_prompt], target)[0]
    answer = scorer.generate_answer(system_prompt, max_new_tokens=max_new_tokens)
    return ScoredPrompt(
        system_prompt=system_prompt,
        score=score,
        logprob_score=logprob_score,
        target_rank=target_rank,
        answer=answer,
    )


def run_baseline(
    scorer: LlamaScorer,
    length: int,
    samples: int,
    batch_size: int,
    seed: int,
    objective: str,
    target: str,
) -> ScoredPrompt:
    rng = random.Random(seed)
    prompts = [""]
    prompts.extend(
        numeric_list(rng.randrange(1000) for _ in range(length))
        for _ in range(samples)
    )

    scored: list[ScoredPrompt] = []
    for batch in batched(prompts, batch_size):
        scores = scorer.score_objective(batch, objective, target)
        scored.extend(ScoredPrompt(prompt, score) for prompt, score in zip(batch, scores))

    best = max(scored, key=lambda item: item.score)
    return score_one(scorer, best.system_prompt, objective, target)


def run_greedy(
    scorer: LlamaScorer,
    length: int,
    batch_size: int,
    objective: str,
    target: str,
) -> tuple[ScoredPrompt, list[GreedyStep]]:
    chosen: list[str] = []
    options = candidate_numbers()
    history: list[GreedyStep] = []

    for position in range(length):
        candidates = [
            ", ".join(chosen + [option])
            for option in options
        ]
        best_prompt = ""
        best_score = -math.inf
        best_option = ""

        for batch in batched(candidates, batch_size):
            scores = scorer.score_objective(batch, objective, target)
            for prompt, score in zip(batch, scores):
                if score > best_score:
                    best_prompt = prompt
                    best_score = score
                    best_option = prompt.split(", ")[-1]

        chosen.append(best_option)
        history.append(
            GreedyStep(
                length=position + 1,
                chosen_number=best_option,
                system_prompt=", ".join(chosen),
                score=best_score,
            )
        )
        print(
            f"greedy position {position + 1}/{length}: "
            f"picked {best_option}, score={best_score:.4f}, prompt={best_prompt}",
            flush=True,
        )

    prompt = ", ".join(chosen)
    return score_one(scorer, prompt, objective, target), history


def write_greedy_curve_csv(
    path: Path,
    baseline: ScoredPrompt,
    history: list[GreedyStep],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "length",
                "method",
                "chosen_number",
                "system_prompt",
                "objective_score",
                "target_logprob",
                "target_rank",
                "answer",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "length": 0,
                "method": "empty_baseline",
                "chosen_number": "",
                "system_prompt": baseline.system_prompt,
                "objective_score": baseline.score,
                "target_logprob": baseline.logprob_score or "",
                "target_rank": baseline.target_rank or "",
                "answer": baseline.answer or "",
            }
        )
        for step in history:
            writer.writerow(
                {
                    "length": step.length,
                    "method": "greedy",
                    "chosen_number": step.chosen_number,
                    "system_prompt": step.system_prompt,
                    "objective_score": step.score,
                    "target_logprob": step.logprob_score or "",
                    "target_rank": step.target_rank or "",
                    "answer": step.answer or "",
                }
            )


def write_greedy_curve_plot(
    path: Path,
    baseline: ScoredPrompt,
    history: list[GreedyStep],
    objective: str,
    target: str,
) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = [step.length for step in history]
    scores = [step.score for step in history]
    baseline_scores = [baseline.score for _ in lengths]
    objective_label = {
        "logprob": f'Log P("{target}")',
        "above-margin": f'{target} logit - logsumexp(tokens above "{target}")',
        "top-margin": f'{target} logit - top non-{target} logit',
        "fixed-margin": f'{target} logit - fixed competitor logsumexp',
    }[objective]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(lengths, scores, marker="o", label=f"Greedy prefix {objective_label}")
    ax.plot(lengths, baseline_scores, linestyle="--", label="Empty baseline")
    ax.set_title("Greedy numeric-list prompt effectiveness")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel(objective_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_greedy_curve(
    scorer: LlamaScorer,
    length: int,
    batch_size: int,
    csv_path: Path,
    plot_path: Path,
    objective: str,
    target: str,
) -> tuple[ScoredPrompt, list[GreedyStep]]:
    baseline = score_one(scorer, "", objective, target)
    greedy, history = run_greedy(
        scorer=scorer,
        length=length,
        batch_size=batch_size,
        objective=objective,
        target=target,
    )
    history_with_answers = [
        GreedyStep(
            length=step.length,
            chosen_number=step.chosen_number,
            system_prompt=step.system_prompt,
            score=step.score,
            logprob_score=scorer.score_target([step.system_prompt], target)[0],
            target_rank=scorer.target_ranks([step.system_prompt], target)[0],
            answer=scorer.generate_answer(step.system_prompt),
        )
        for step in history
    ]
    write_greedy_curve_csv(csv_path, baseline, history_with_answers)
    write_greedy_curve_plot(plot_path, baseline, history_with_answers, objective, target)
    print(f"\nwrote CSV: {csv_path}")
    print(f"wrote plot: {plot_path}")
    return greedy, history_with_answers


def tournament_select(
    population: list[tuple[int, ...]],
    scores: list[float],
    rng: random.Random,
    tournament_size: int,
) -> tuple[int, ...]:
    indexes = rng.sample(range(len(population)), k=tournament_size)
    winner = max(indexes, key=lambda index: scores[index])
    return population[winner]


def crossover(
    left: tuple[int, ...],
    right: tuple[int, ...],
    rng: random.Random,
    mode: str,
) -> tuple[int, ...]:
    if len(left) != len(right):
        raise ValueError("Cannot crossover genomes with different lengths.")
    if len(left) < 2:
        return left
    if mode == "uniform":
        return tuple(a if rng.random() < 0.5 else b for a, b in zip(left, right))
    if mode == "one-point":
        point = rng.randrange(1, len(left))
        return left[:point] + right[point:]
    raise ValueError(f"Unknown crossover mode: {mode}")


def mutate(
    genome: tuple[int, ...],
    rng: random.Random,
    mutation_rate: float,
) -> tuple[int, ...]:
    mutated = list(genome)
    changed = False
    for index in range(len(mutated)):
        if rng.random() < mutation_rate:
            mutated[index] = rng.randrange(1000)
            changed = True
    if not changed and mutation_rate > 0:
        mutated[rng.randrange(len(mutated))] = rng.randrange(1000)
    return tuple(mutated)


def evaluate_prompts(
    scorer: LlamaScorer,
    prompts: list[str],
    batch_size: int,
    objective: str,
    target: str,
) -> list[float]:
    scores: list[float] = []
    for batch in batched(prompts, batch_size):
        scores.extend(scorer.score_objective(batch, objective, target))
    return scores


def _ga_score_worker(
    worker_index: int,
    device: str,
    model_name: str,
    max_seq_length: int,
    load_in_4bit: bool,
    dtype: str | None,
    question: str,
    competitors: list[str],
    batch_size: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    if device:
        os.environ["CUDA_VISIBLE_DEVICES"] = device
    load_dotenv()
    scorer = LlamaScorer(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=dtype,
        question=question,
    )
    if competitors:
        scorer.set_fixed_competitors(competitors)

    while True:
        task = task_queue.get()
        if task is None:
            break
        task_id, prompts, objective, target = task
        try:
            scores = evaluate_prompts(scorer, prompts, batch_size, objective, target)
            result_queue.put((task_id, worker_index, scores, None))
        except Exception as exc:  # pragma: no cover - exercised by worker process.
            result_queue.put((task_id, worker_index, [], repr(exc)))


class DistributedPromptEvaluator:
    def __init__(
        self,
        devices: list[str],
        model_name: str,
        max_seq_length: int,
        load_in_4bit: bool,
        dtype: str | None,
        question: str,
        competitors: list[str],
        batch_size: int,
        task_size: int | None,
        show_progress: bool,
    ) -> None:
        self.devices = devices
        self.task_size = task_size
        self.show_progress = show_progress
        self.task_queue: mp.Queue = mp.Queue()
        self.result_queue: mp.Queue = mp.Queue()
        self.processes: list[mp.Process] = []
        context = mp.get_context("spawn")
        self.task_queue = context.Queue()
        self.result_queue = context.Queue()
        for worker_index, device in enumerate(devices):
            process = context.Process(
                target=_ga_score_worker,
                args=(
                    worker_index,
                    device,
                    model_name,
                    max_seq_length,
                    load_in_4bit,
                    dtype,
                    question,
                    competitors,
                    batch_size,
                    self.task_queue,
                    self.result_queue,
                ),
            )
            process.start()
            self.processes.append(process)

    def close(self) -> None:
        for _process in self.processes:
            self.task_queue.put(None)
        for process in self.processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    def __enter__(self) -> DistributedPromptEvaluator:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def score_prompts(self, prompts: list[str], objective: str, target: str) -> list[float]:
        if not prompts:
            return []
        chunks: list[tuple[int, list[str]]] = []
        worker_count = len(self.processes)
        chunk_size = self.task_size or math.ceil(len(prompts) / worker_count)
        chunk_size = max(1, chunk_size)
        for task_id, start in enumerate(range(0, len(prompts), chunk_size)):
            chunk = prompts[start : start + chunk_size]
            chunks.append((task_id, chunk))
            self.task_queue.put((task_id, chunk, objective, target))

        results: dict[int, list[float]] = {}
        errors: list[str] = []
        deadline = time.monotonic() + 60 * 60 * 24
        started = time.monotonic()
        while len(results) < len(chunks):
            try:
                task_id, worker_index, scores, error = self.result_queue.get(timeout=5)
            except Empty:
                if any(not process.is_alive() for process in self.processes):
                    raise RuntimeError("A distributed GA worker exited unexpectedly.")
                if time.monotonic() > deadline:
                    raise TimeoutError("Timed out waiting for distributed GA workers.")
                continue
            if error:
                errors.append(f"worker {worker_index}: {error}")
            else:
                results[task_id] = scores
                if self.show_progress:
                    done = len(results)
                    elapsed = time.monotonic() - started
                    print(
                        f"distributed score: {done}/{len(chunks)} chunks "
                        f"({done * 100 / len(chunks):.1f}%) in {elapsed:.1f}s",
                        flush=True,
                    )
        if errors:
            raise RuntimeError("; ".join(errors))
        combined: list[float] = []
        for task_id, _chunk in chunks:
            combined.extend(results[task_id])
        return combined


def write_ga_csv(path: Path, history: list[GAStep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "generation",
                "system_prompt",
                "objective_score",
                "target_logprob",
                "target_rank",
                "answer",
            ],
        )
        writer.writeheader()
        for step in history:
            writer.writerow(
                {
                    "generation": step.generation,
                    "system_prompt": step.system_prompt,
                    "objective_score": step.score,
                    "target_logprob": step.logprob_score or "",
                    "target_rank": step.target_rank or "",
                    "answer": step.answer or "",
                }
            )


def write_ga_plot(path: Path, history: list[GAStep], objective: str, target: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    generations = [step.generation for step in history]
    scores = [step.score for step in history]
    objective_label = {
        "logprob": f'Log P("{target}")',
        "above-margin": f'{target} logit - logsumexp(tokens above "{target}")',
        "top-margin": f'{target} logit - top non-{target} logit',
        "fixed-margin": f'{target} logit - fixed competitor logsumexp',
    }[objective]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(generations, scores, marker="o", label=f"Best population {objective_label}")
    ax.set_title("Genetic numeric-list prompt optimization")
    ax.set_xlabel("Generation")
    ax.set_ylabel(objective_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_igcg_csv(path: Path, history: list[IGCGStep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "system_prompt",
                "objective_score",
                "target_logprob",
                "target_rank",
                "answer",
            ],
        )
        writer.writeheader()
        for step in history:
            writer.writerow(
                {
                    "step": step.step,
                    "system_prompt": step.system_prompt,
                    "objective_score": step.score,
                    "target_logprob": step.logprob_score or "",
                    "target_rank": step.target_rank or "",
                    "answer": step.answer or "",
                }
            )


def write_igcg_plot(path: Path, history: list[IGCGStep], objective: str, target: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [step.step for step in history]
    scores = [step.score for step in history]
    objective_label = {
        "logprob": f'Log P("{target}")',
        "above-margin": f'{target} logit - logsumexp(tokens above "{target}")',
        "top-margin": f'{target} logit - top non-{target} logit',
        "fixed-margin": f'{target} logit - fixed competitor logsumexp',
    }[objective]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, scores, marker="o", label=f"I-GCG {objective_label}")
    ax.set_title("Restricted-vocab I-GCG numeric-list prompt optimization")
    ax.set_xlabel("Step")
    ax.set_ylabel(objective_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_adc_csv(path: Path, history: list[ADCStep]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "position",
                "chosen_number",
                "system_prompt",
                "objective_score",
                "target_logprob",
                "target_rank",
                "answer",
            ],
        )
        writer.writeheader()
        for step in history:
            writer.writerow(
                {
                    "step": step.step,
                    "position": step.position,
                    "chosen_number": step.chosen_number,
                    "system_prompt": step.system_prompt,
                    "objective_score": step.score,
                    "target_logprob": step.logprob_score or "",
                    "target_rank": step.target_rank or "",
                    "answer": step.answer or "",
                }
            )


def write_adc_plot(path: Path, history: list[ADCStep], objective: str, target: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    steps = [step.step for step in history]
    scores = [step.score for step in history]
    objective_label = {
        "logprob": f'Log P("{target}")',
        "above-margin": f'{target} logit - logsumexp(tokens above "{target}")',
        "top-margin": f'{target} logit - top non-{target} logit',
        "fixed-margin": f'{target} logit - fixed competitor logsumexp',
    }[objective]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, scores, marker="o", label=f"ADC {objective_label}")
    ax.set_title("Restricted-vocab adaptive discrete coordinate descent")
    ax.set_xlabel("Coordinate update")
    ax.set_ylabel(objective_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def sample_igcg_candidates(
    genome: tuple[int, ...],
    grad: torch.Tensor,
    token_id_to_value: dict[int, int],
    allowed_token_ids: list[int],
    search_width: int,
    topk: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    allowed = torch.tensor(allowed_token_ids, dtype=torch.long, device=grad.device)
    restricted_grad = grad[:, allowed]
    k = min(topk, restricted_grad.shape[1])
    top_allowed_offsets = (-restricted_grad).topk(k, dim=1).indices.detach().cpu().tolist()
    candidates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    positions = list(range(len(genome)))

    for sample_index in range(search_width):
        position = positions[sample_index % len(positions)]
        token_offset = rng.choice(top_allowed_offsets[position])
        token_id = allowed_token_ids[token_offset]
        value = token_id_to_value[token_id]
        candidate = list(genome)
        candidate[position] = value
        candidate_tuple = tuple(candidate)
        if candidate_tuple not in seen and candidate_tuple != genome:
            candidates.append(candidate_tuple)
            seen.add(candidate_tuple)

    return candidates


def merge_top_igcg_candidates(
    current: tuple[int, ...],
    ranked_candidates: list[tuple[int, ...]],
    top_k: int,
) -> list[tuple[int, ...]]:
    merged = list(current)
    merged_candidates: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in ranked_candidates[:top_k]:
        for index, (old_value, new_value) in enumerate(zip(current, candidate)):
            if old_value != new_value:
                merged[index] = new_value
        merged_tuple = tuple(merged)
        if merged_tuple not in seen and merged_tuple != current:
            merged_candidates.append(merged_tuple)
            seen.add(merged_tuple)
    return merged_candidates


def run_igcg(
    scorer: LlamaScorer,
    length: int,
    steps: int,
    search_width: int,
    batch_size: int,
    objective: str,
    target: str,
    seed: int,
    topk: int,
    merge_top_k: int,
    csv_path: Path,
    plot_path: Path,
    init_prompt: str | None,
) -> ScoredPrompt:
    if length < 1:
        raise ValueError("--length must be at least 1 for I-GCG.")
    if search_width < 1:
        raise ValueError("--search-width must be at least 1 for I-GCG.")
    if merge_top_k < 1:
        raise ValueError("--merge-top-k must be at least 1 for I-GCG.")

    rng = random.Random(seed)
    allowed = scorer.allowed_numeric_token_ids()
    if not allowed:
        raise ValueError("No single-token 3-digit numeric strings are available.")
    token_id_to_value = {token_id: int(text) for text, token_id in allowed.items()}
    allowed_token_ids = list(token_id_to_value)
    allowed_values = [int(text) for text in allowed]

    if init_prompt:
        values = [int(part.strip()) for part in init_prompt.split(",")]
        if len(values) != length:
            raise ValueError("--init-prompt must contain exactly --length numbers.")
        genome = tuple(values)
    else:
        genome = tuple(rng.choice(allowed_values) for _ in range(length))

    history: list[IGCGStep] = []
    current = score_one(scorer, genome_to_prompt(genome), objective, target)
    best = current
    history.append(
        IGCGStep(
            step=0,
            system_prompt=current.system_prompt,
            score=current.score,
            logprob_score=current.logprob_score,
            target_rank=current.target_rank,
            answer=current.answer,
        )
    )
    print(
        f"igcg step 0/{steps}: score={current.score:.4f}, "
        f"logprob={current.logprob_score:.4f}, rank={current.target_rank}, "
        f"answer={current.answer!r}, prompt={current.system_prompt}",
        flush=True,
    )

    for step in range(1, steps + 1):
        grad, prompt_ids, control_positions, prompt_len = scorer.objective_token_gradients(
            genome,
            objective,
            target,
        )
        target_ids = scorer.target_ids(target)
        candidate_genomes = sample_igcg_candidates(
            genome=genome,
            grad=grad,
            token_id_to_value=token_id_to_value,
            allowed_token_ids=allowed_token_ids,
            search_width=search_width,
            topk=topk,
            rng=rng,
        )
        if not candidate_genomes:
            break

        candidate_ids: list[list[int]] = []
        for candidate in candidate_genomes:
            ids = list(prompt_ids)
            for position, value in zip(control_positions, candidate):
                ids[position] = allowed[f"{value:03d}"]
            candidate_ids.append(ids + target_ids)

        scores = scorer.evaluate_candidate_ids(
            candidate_ids=candidate_ids,
            prompt_len=prompt_len,
            target_ids=target_ids,
            batch_size=batch_size,
            objective=objective,
        )
        ranked = [
            candidate
            for candidate, _score in sorted(
                zip(candidate_genomes, scores),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        merged = merge_top_igcg_candidates(genome, ranked, merge_top_k)
        evaluation_genomes = ranked[:1] + merged
        evaluation_ids: list[list[int]] = []
        for candidate in evaluation_genomes:
            ids = list(prompt_ids)
            for position, value in zip(control_positions, candidate):
                ids[position] = allowed[f"{value:03d}"]
            evaluation_ids.append(ids + target_ids)
        evaluation_scores = scorer.evaluate_candidate_ids(
            candidate_ids=evaluation_ids,
            prompt_len=prompt_len,
            target_ids=target_ids,
            batch_size=batch_size,
            objective=objective,
        )
        best_index = max(range(len(evaluation_scores)), key=evaluation_scores.__getitem__)
        if evaluation_scores[best_index] >= current.score:
            genome = evaluation_genomes[best_index]

        current = score_one(scorer, genome_to_prompt(genome), objective, target)
        if current.score > best.score:
            best = current
        history.append(
            IGCGStep(
                step=step,
                system_prompt=current.system_prompt,
                score=current.score,
                logprob_score=current.logprob_score,
                target_rank=current.target_rank,
                answer=current.answer,
            )
        )
        print(
            f"igcg step {step}/{steps}: score={current.score:.4f}, "
            f"logprob={current.logprob_score:.4f}, rank={current.target_rank}, "
            f"answer={current.answer!r}, prompt={current.system_prompt}",
            flush=True,
        )

    write_igcg_csv(csv_path, history)
    write_igcg_plot(plot_path, history, objective, target)
    print(f"\nwrote CSV: {csv_path}")
    print(f"wrote plot: {plot_path}")
    return best


def parse_numeric_prompt(init_prompt: str, length: int) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in init_prompt.split(","))
    if len(values) != length:
        raise ValueError("--init-prompt must contain exactly --length numbers.")
    return values


def parse_numeric_prompt_text(system_prompt: str, length: int) -> tuple[int, ...]:
    return parse_numeric_prompt(system_prompt, length)


def read_population_csv(path: Path, length: int) -> list[tuple[int, ...]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if "system_prompt" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a system_prompt column.")
        return [
            parse_numeric_prompt_text(row["system_prompt"], length)
            for row in reader
            if row.get("system_prompt")
        ]


def write_population_csv(
    path: Path,
    ranked: list[tuple[tuple[int, ...], str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["rank", "system_prompt", "objective_score"],
        )
        writer.writeheader()
        for rank, (_genome, prompt, score) in enumerate(ranked, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "system_prompt": prompt,
                    "objective_score": score,
                }
            )


def run_adc(
    scorer: LlamaScorer,
    length: int,
    steps: int,
    batch_size: int,
    objective: str,
    target: str,
    seed: int,
    csv_path: Path,
    plot_path: Path,
    init_prompt: str | None,
    shuffle_positions: bool,
    rerank_top_k: int,
) -> ScoredPrompt:
    if length < 1:
        raise ValueError("--length must be at least 1 for ADC.")
    if steps < 1:
        raise ValueError("--steps must be at least 1 for ADC.")
    if rerank_top_k < 1:
        raise ValueError("--adc-rerank-top-k must be at least 1.")

    rng = random.Random(seed)
    allowed_values = list(range(1000))
    if init_prompt:
        genome = parse_numeric_prompt(init_prompt, length)
    else:
        genome = tuple(rng.choice(allowed_values) for _ in range(length))

    history: list[ADCStep] = []
    current = score_one(scorer, genome_to_prompt(genome), objective, target)
    best = current
    history.append(
        ADCStep(
            step=0,
            position=-1,
            chosen_number="",
            system_prompt=current.system_prompt,
            score=current.score,
            logprob_score=current.logprob_score,
            target_rank=current.target_rank,
            answer=current.answer,
        )
    )
    print(
        f"adc step 0/{steps}: score={current.score:.4f}, "
        f"logprob={current.logprob_score:.4f}, rank={current.target_rank}, "
        f"answer={current.answer!r}, prompt={current.system_prompt}",
        flush=True,
    )

    positions = list(range(length))
    for step in range(1, steps + 1):
        if (step - 1) % length == 0:
            positions = list(range(length))
            if shuffle_positions:
                rng.shuffle(positions)
        position = positions[(step - 1) % length]

        candidates: list[tuple[int, ...]] = [genome]
        for value in allowed_values:
            if value == genome[position]:
                continue
            candidate = list(genome)
            candidate[position] = value
            candidates.append(tuple(candidate))
        prompts = [genome_to_prompt(candidate) for candidate in candidates]
        scores = evaluate_prompts(scorer, prompts, batch_size, objective, target)
        ranked_indexes = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        rerank_indexes = sorted(set([0] + ranked_indexes[:rerank_top_k]))
        rerank_prompts = [prompts[index] for index in rerank_indexes]
        rerank_scores = [
            scorer.score_objective([prompt], objective, target)[0]
            for prompt in rerank_prompts
        ]
        best_rerank_offset = max(range(len(rerank_scores)), key=rerank_scores.__getitem__)
        genome = candidates[rerank_indexes[best_rerank_offset]]

        current = score_one(scorer, genome_to_prompt(genome), objective, target)
        if current.score > best.score:
            best = current
        history.append(
            ADCStep(
                step=step,
                position=position,
                chosen_number=f"{genome[position]:03d}",
                system_prompt=current.system_prompt,
                score=current.score,
                logprob_score=current.logprob_score,
                target_rank=current.target_rank,
                answer=current.answer,
            )
        )
        print(
            f"adc step {step}/{steps}: pos={position}, value={genome[position]:03d}, "
            f"score={current.score:.4f}, logprob={current.logprob_score:.4f}, "
            f"rank={current.target_rank}, answer={current.answer!r}, "
            f"prompt={current.system_prompt}",
            flush=True,
        )

    write_adc_csv(csv_path, history)
    write_adc_plot(plot_path, history, objective, target)
    print(f"\nwrote CSV: {csv_path}")
    print(f"wrote plot: {plot_path}")
    return best


def run_ga(
    scorer: LlamaScorer | None,
    length: int,
    population_size: int,
    generations: int,
    batch_size: int,
    objective: str,
    target: str,
    seed: int,
    elite_count: int,
    mutation_rate: float,
    tournament_size: int,
    crossover_mode: str,
    csv_path: Path,
    plot_path: Path,
    population_path: Path | None,
    init_population_path: Path | None,
    report_every: int,
    generate_during_search: bool,
    cache_scores: bool,
    final_population_size: int | None,
    distributed_evaluator: DistributedPromptEvaluator | None = None,
) -> ScoredPrompt:
    if population_size < 2:
        raise ValueError("--population-size must be at least 2.")
    if elite_count < 1 or elite_count >= population_size:
        raise ValueError("--elite-count must be >= 1 and smaller than --population-size.")
    if tournament_size < 2 or tournament_size > population_size:
        raise ValueError("--tournament-size must be between 2 and --population-size.")
    if length < 1:
        raise ValueError("--length must be at least 1 for GA.")
    if report_every < 1:
        raise ValueError("--report-every must be at least 1.")
    if scorer is None and distributed_evaluator is None:
        raise ValueError("GA requires either a local scorer or distributed evaluator.")
    if final_population_size is not None:
        if final_population_size < 2:
            raise ValueError("--final-population-size must be at least 2.")
        if final_population_size > population_size:
            raise ValueError("--final-population-size cannot exceed --population-size.")

    rng = random.Random(seed)
    population: list[tuple[int, ...]] = []
    if init_population_path is not None:
        population = read_population_csv(init_population_path, length)
        if not population:
            raise ValueError(f"No genomes found in {init_population_path}.")
        population = population[:population_size]
    while len(population) < population_size:
        population.append(tuple(rng.randrange(1000) for _ in range(length)))
    history: list[GAStep] = []
    best_prompt = ""
    best_score = -math.inf
    score_cache: dict[tuple[int, ...], float] = {}
    final_ranked: list[tuple[tuple[int, ...], str, float]] = []

    for generation in range(generations + 1):
        if final_population_size is None or generations == 0:
            target_population_size = population_size
        else:
            progress = generation / generations
            target_population_size = round(
                population_size
                - (population_size - final_population_size) * progress
            )
            target_population_size = max(final_population_size, target_population_size)
        if len(population) > target_population_size:
            population = population[:target_population_size]

        scores_by_genome: dict[tuple[int, ...], float] = {}
        missing = [
            genome
            for genome in population
            if not cache_scores or genome not in score_cache
        ]
        if missing:
            missing_prompts = [genome_to_prompt(genome) for genome in missing]
            if distributed_evaluator is not None:
                missing_scores = distributed_evaluator.score_prompts(
                    missing_prompts,
                    objective,
                    target,
                )
            else:
                assert scorer is not None
                missing_scores = evaluate_prompts(
                    scorer,
                    missing_prompts,
                    batch_size,
                    objective,
                    target,
                )
            if cache_scores:
                score_cache.update(zip(missing, missing_scores))
            else:
                scores_by_genome.update(zip(missing, missing_scores))

        if cache_scores:
            scores = [score_cache[genome] for genome in population]
        else:
            scores = [scores_by_genome[genome] for genome in population]
        prompts = [genome_to_prompt(genome) for genome in population]
        ranked = sorted(
            zip(population, prompts, scores),
            key=lambda item: item[2],
            reverse=True,
        )
        final_ranked = ranked
        generation_best_genome, generation_best_prompt, generation_best_score = ranked[0]
        if generation_best_score > best_score:
            best_prompt = generation_best_prompt
            best_score = generation_best_score

        should_report_details = (
            scorer is not None
            and (generation % report_every == 0 or generation == generations)
        )
        best_logprob = None
        best_rank = None
        best_answer = None
        if should_report_details:
            best_logprob = scorer.score_target([generation_best_prompt], target)[0]
            best_rank = scorer.target_ranks([generation_best_prompt], target)[0]
            if generate_during_search:
                best_answer = scorer.generate_answer(generation_best_prompt)
        history.append(
            GAStep(
                generation=generation,
                system_prompt=generation_best_prompt,
                score=generation_best_score,
                logprob_score=best_logprob,
                target_rank=best_rank,
                answer=best_answer,
            )
        )
        if generation % report_every == 0 or generation == generations:
            details = ""
            if best_logprob is not None:
                details += f", logprob={best_logprob:.4f}"
            if best_rank is not None:
                details += f", rank={best_rank}"
            if best_answer is not None:
                details += f", answer={best_answer!r}"
            cache_details = f", cache={len(score_cache)}" if cache_scores else ""
            print(
                f"ga generation {generation}/{generations}: "
                f"pop={len(population)}, score={generation_best_score:.4f}"
                f"{details}{cache_details}, "
                f"prompt={generation_best_prompt}",
                flush=True,
            )

        if generation == generations:
            break

        next_size = population_size
        if final_population_size is not None:
            next_progress = (generation + 1) / generations
            next_size = round(
                population_size
                - (population_size - final_population_size) * next_progress
            )
            next_size = max(final_population_size, next_size)
        effective_elite_count = min(elite_count, next_size - 1)
        next_population = [item[0] for item in ranked[:effective_elite_count]]
        seen = set(next_population)
        while len(next_population) < next_size:
            parent_a = tournament_select(population, scores, rng, tournament_size)
            parent_b = tournament_select(population, scores, rng, tournament_size)
            child = crossover(parent_a, parent_b, rng, crossover_mode)
            child = mutate(child, rng, mutation_rate)
            if child in seen:
                child = mutate(child, rng, 1.0 / length)
            next_population.append(child)
            seen.add(child)
        population = next_population

    write_ga_csv(csv_path, history)
    write_ga_plot(plot_path, history, objective, target)
    if population_path is not None:
        write_population_csv(population_path, final_ranked)
    print(f"\nwrote CSV: {csv_path}")
    print(f"wrote plot: {plot_path}")
    if population_path is not None:
        print(f"wrote population: {population_path}")
    if scorer is not None:
        return score_one(scorer, best_prompt, objective, target)
    return ScoredPrompt(system_prompt=best_prompt, score=best_score)


def print_result(label: str, result: ScoredPrompt) -> None:
    shown_prompt = result.system_prompt if result.system_prompt else "<empty>"
    print(f"\n[{label}]")
    print(f"system_prompt: {shown_prompt}")
    print(f"objective_score: {result.score:.4f}")
    if result.logprob_score is not None and result.logprob_score != result.score:
        print(f'target_logprob: {result.logprob_score:.4f}')
    if result.target_rank is not None:
        print(f"target_rank: {result.target_rank}")
    print(f"generated_answer: {result.answer}")


def write_score_csv(path: Path, result: ScoredPrompt) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "system_prompt",
                "objective_score",
                "target_logprob",
                "target_rank",
                "answer",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "system_prompt": result.system_prompt,
                "objective_score": result.score,
                "target_logprob": result.logprob_score or "",
                "target_rank": result.target_rank or "",
                "answer": result.answer or "",
            }
        )


def write_transcript_csv(path: Path, results: list[TranscriptResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "rank",
                "row_indices",
                "system_prompt",
                "objective_score",
                "target_logprob",
                "target_rank",
                "answer",
                "animal_ranking",
            ],
        )
        writer.writeheader()
        for rank, item in enumerate(
            sorted(results, key=lambda transcript: transcript.result.score, reverse=True),
            start=1,
        ):
            result = item.result
            writer.writerow(
                {
                    "rank": rank,
                    "row_indices": ",".join(str(index) for index in item.row_indices),
                    "system_prompt": result.system_prompt,
                    "objective_score": result.score,
                    "target_logprob": result.logprob_score or "",
                    "target_rank": result.target_rank or "",
                    "answer": result.answer or "",
                    "animal_ranking": format_animal_ranking(item.animal_scores),
                }
            )


def run_transcript(
    scorer: LlamaScorer,
    dataset_name: str,
    dataset_split: str,
    transcript_rows: int,
    row_indices_text: str,
    search_samples: int,
    objective: str,
    target: str,
    seed: int,
    system_prompt: str,
    max_new_tokens: int,
    animals: list[str],
    csv_path: Path,
) -> TranscriptResult:
    rows = load_transcript_rows(dataset_name, dataset_split)
    if transcript_rows < 1:
        raise ValueError("--transcript-rows must be at least 1.")
    if transcript_rows > len(rows):
        raise ValueError(
            f"--transcript-rows={transcript_rows} exceeds dataset size {len(rows)} for {dataset_split!r}."
        )

    rng = random.Random(seed)
    candidate_indices: list[tuple[int, ...]] = []
    if row_indices_text:
        row_indices = tuple(parse_row_indices(row_indices_text))
        if len(row_indices) != transcript_rows:
            raise ValueError("--transcript-row-indices length must match --transcript-rows.")
        candidate_indices.append(row_indices)
    else:
        candidate_indices.append(tuple(range(transcript_rows)))

    for _ in range(search_samples):
        candidate_indices.append(tuple(sorted(rng.sample(range(len(rows)), transcript_rows))))

    results: list[TranscriptResult] = []
    seen: set[tuple[int, ...]] = set()
    for row_indices in candidate_indices:
        if row_indices in seen:
            continue
        seen.add(row_indices)
        scorer.set_transcript_messages(transcript_messages_from_rows(rows, row_indices))
        result = score_one(
            scorer=scorer,
            system_prompt=system_prompt,
            objective=objective,
            target=target,
            max_new_tokens=max_new_tokens,
        )
        animal_scores = score_animal_list(scorer, system_prompt, animals)
        results.append(
            TranscriptResult(
                row_indices=row_indices,
                result=result,
                animal_scores=animal_scores,
            )
        )
        print(
            "transcript rows="
            f"{','.join(str(index) for index in row_indices)} "
            f"score={result.score:.4f} logprob={result.logprob_score:.4f} "
            f"rank={result.target_rank} answer={result.answer!r}"
        )
        print(f"  animals: {format_animal_ranking(animal_scores)}")

    write_transcript_csv(csv_path, results)
    return max(results, key=lambda item: item.result.score)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=[
            "baseline",
            "greedy",
            "both",
            "greedy-curve",
            "ga",
            "igcg",
            "adc",
            "score",
            "transcript",
        ],
        default="both",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--target", default=TARGET)
    parser.add_argument(
        "--animals",
        default=DEFAULT_ANIMALS,
        help="Comma-separated animals to score for transcript runs.",
    )
    parser.add_argument("--length", type=int, default=5)
    parser.add_argument("--baseline-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dtype", default=None, help="Optional torch dtype name, e.g. float16.")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument(
        "--objective",
        choices=["logprob", "above-margin", "top-margin", "fixed-margin"],
        default="logprob",
        help=(
            "logprob maximizes log P(target). above-margin maximizes the target "
            "first-token logit minus logsumexp of all logits currently above it. "
            "top-margin maximizes target first-token logit minus the best non-target logit. "
            "fixed-margin maximizes target first-token logit minus fixed competitors."
        ),
    )
    parser.add_argument(
        "--competitors",
        default="",
        help='Pipe-separated fixed-margin competitors, e.g. "Dog|I|Cat|D|Human|No|L".',
    )
    parser.add_argument("--csv-path", type=Path, default=Path("outputs/greedy_curve.csv"))
    parser.add_argument("--plot-path", type=Path, default=Path("outputs/greedy_curve.png"))
    parser.add_argument(
        "--population-path",
        type=Path,
        default=None,
        help="For GA, write the final ranked population to this CSV.",
    )
    parser.add_argument(
        "--init-population-path",
        type=Path,
        default=None,
        help="For GA, seed the initial population from a saved population CSV.",
    )
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument(
        "--final-population-size",
        type=int,
        default=None,
        help="For GA, linearly taper population from --population-size to this value.",
    )
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--mutation-rate", type=float, default=0.08)
    parser.add_argument("--tournament-size", type=int, default=5)
    parser.add_argument("--crossover", choices=["uniform", "one-point"], default="uniform")
    parser.add_argument(
        "--ga-workers",
        type=int,
        default=1,
        help="Number of persistent GA scoring workers. Use one per GPU for multi-GPU runs.",
    )
    parser.add_argument(
        "--ga-task-size",
        type=int,
        default=0,
        help="For distributed GA, prompts per queue task. 0 splits once per worker.",
    )
    parser.add_argument(
        "--ga-progress",
        action="store_true",
        help="For distributed GA, print chunk-level scoring progress.",
    )
    parser.add_argument(
        "--cuda-devices",
        default="",
        help='Comma-separated CUDA device IDs for GA workers, e.g. "0,1,2,3".',
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=1,
        help="For GA, print/report detailed best-prompt metrics every N generations.",
    )
    parser.add_argument(
        "--no-generate-during-search",
        action="store_true",
        help="For GA, skip generated answer diagnostics during the search loop.",
    )
    parser.add_argument(
        "--no-score-cache",
        action="store_true",
        help="For GA, disable genome objective score caching.",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--search-width", type=int, default=256)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--merge-top-k", type=int, default=7)
    parser.add_argument(
        "--no-shuffle-positions",
        action="store_true",
        help="For ADC, sweep positions in fixed order instead of shuffling each pass.",
    )
    parser.add_argument(
        "--adc-rerank-top-k",
        type=int,
        default=16,
        help="For ADC, single-prompt rerank this many top batched coordinate edits before accepting.",
    )
    parser.add_argument(
        "--init-prompt",
        default=None,
        help='Optional comma-separated numeric list, e.g. "500, 942, 236".',
    )
    parser.add_argument(
        "--transcript-dataset",
        default=DEFAULT_TRANSCRIPT_DATASET,
        help="Hugging Face dataset containing prompt/completion rows for in-context transcript tests.",
    )
    parser.add_argument(
        "--transcript-split",
        default="train",
        help='Dataset split for transcript rows, e.g. "train" or "train[:1000]".',
    )
    parser.add_argument(
        "--transcript-rows",
        type=int,
        default=8,
        help="Number of dataset prompt/completion rows to place before the final question.",
    )
    parser.add_argument(
        "--transcript-row-indices",
        default="",
        help="Comma-separated dataset row indices to use. If omitted, uses the first N rows.",
    )
    parser.add_argument(
        "--transcript-search-samples",
        type=int,
        default=0,
        help="Number of random row subsets to score in addition to the explicit/default subset.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    competitors = args.competitors.split("|") if args.competitors else []
    use_distributed_ga = args.method == "ga" and args.ga_workers > 1
    scorer = None
    if not use_distributed_ga:
        scorer = LlamaScorer(
            model_name=args.model,
            max_seq_length=args.max_seq_length,
            load_in_4bit=not args.no_4bit,
            dtype=args.dtype,
            question=args.question,
        )
        if competitors:
            scorer.set_fixed_competitors(competitors)

    if args.method in {"baseline", "both"}:
        assert scorer is not None
        baseline = run_baseline(
            scorer=scorer,
            length=args.length,
            samples=args.baseline_samples,
            batch_size=args.batch_size,
            seed=args.seed,
            objective=args.objective,
            target=args.target,
        )
        print_result("baseline", baseline)

    if args.method in {"greedy", "both"}:
        assert scorer is not None
        greedy, _ = run_greedy(
            scorer=scorer,
            length=args.length,
            batch_size=args.batch_size,
            objective=args.objective,
            target=args.target,
        )
        print_result("greedy", greedy)

    if args.method == "greedy-curve":
        assert scorer is not None
        greedy, _ = run_greedy_curve(
            scorer=scorer,
            length=args.length,
            batch_size=args.batch_size,
            csv_path=args.csv_path,
            plot_path=args.plot_path,
            objective=args.objective,
            target=args.target,
        )
        print_result("greedy", greedy)

    if args.method == "ga":
        if use_distributed_ga:
            devices = (
                [device.strip() for device in args.cuda_devices.split(",") if device.strip()]
                if args.cuda_devices
                else [str(index) for index in range(args.ga_workers)]
            )
            if len(devices) != args.ga_workers:
                raise ValueError("--cuda-devices length must match --ga-workers.")
            with DistributedPromptEvaluator(
                devices=devices,
                model_name=args.model,
                max_seq_length=args.max_seq_length,
                load_in_4bit=not args.no_4bit,
                dtype=args.dtype,
                question=args.question,
                competitors=competitors,
                batch_size=args.batch_size,
                task_size=args.ga_task_size or None,
                show_progress=args.ga_progress,
            ) as evaluator:
                ga = run_ga(
                    scorer=None,
                    length=args.length,
                    population_size=args.population_size,
                    generations=args.generations,
                    batch_size=args.batch_size,
                    objective=args.objective,
                    target=args.target,
                    seed=args.seed,
                    elite_count=args.elite_count,
                    mutation_rate=args.mutation_rate,
                    tournament_size=args.tournament_size,
                    crossover_mode=args.crossover,
                    csv_path=args.csv_path,
                    plot_path=args.plot_path,
                    population_path=args.population_path,
                    init_population_path=args.init_population_path,
                    report_every=args.report_every,
                    generate_during_search=not args.no_generate_during_search,
                    cache_scores=not args.no_score_cache,
                    final_population_size=args.final_population_size,
                    distributed_evaluator=evaluator,
                )
        else:
            assert scorer is not None
            ga = run_ga(
                scorer=scorer,
                length=args.length,
                population_size=args.population_size,
                generations=args.generations,
                batch_size=args.batch_size,
                objective=args.objective,
                target=args.target,
                seed=args.seed,
                elite_count=args.elite_count,
                mutation_rate=args.mutation_rate,
                tournament_size=args.tournament_size,
                crossover_mode=args.crossover,
                csv_path=args.csv_path,
                plot_path=args.plot_path,
                population_path=args.population_path,
                init_population_path=args.init_population_path,
                report_every=args.report_every,
                generate_during_search=not args.no_generate_during_search,
                cache_scores=not args.no_score_cache,
                final_population_size=args.final_population_size,
            )
        print_result("ga", ga)

    if args.method == "igcg":
        assert scorer is not None
        igcg = run_igcg(
            scorer=scorer,
            length=args.length,
            steps=args.steps,
            search_width=args.search_width,
            batch_size=args.batch_size,
            objective=args.objective,
            target=args.target,
            seed=args.seed,
            topk=args.topk,
            merge_top_k=args.merge_top_k,
            csv_path=args.csv_path,
            plot_path=args.plot_path,
            init_prompt=args.init_prompt,
        )
        print_result("igcg", igcg)

    if args.method == "adc":
        assert scorer is not None
        adc = run_adc(
            scorer=scorer,
            length=args.length,
            steps=args.steps,
            batch_size=args.batch_size,
            objective=args.objective,
            target=args.target,
            seed=args.seed,
            csv_path=args.csv_path,
            plot_path=args.plot_path,
            init_prompt=args.init_prompt,
            shuffle_positions=not args.no_shuffle_positions,
            rerank_top_k=args.adc_rerank_top_k,
        )
        print_result("adc", adc)

    if args.method == "score":
        assert scorer is not None
        if args.init_prompt is None:
            raise ValueError("--method score requires --init-prompt.")
        result = score_one(
            scorer,
            args.init_prompt,
            args.objective,
            args.target,
            max_new_tokens=args.max_new_tokens,
        )
        write_score_csv(args.csv_path, result)
        print(f"\nwrote CSV: {args.csv_path}")
        print_result("score", result)

    if args.method == "transcript":
        assert scorer is not None
        transcript = run_transcript(
            scorer=scorer,
            dataset_name=args.transcript_dataset,
            dataset_split=args.transcript_split,
            transcript_rows=args.transcript_rows,
            row_indices_text=args.transcript_row_indices,
            search_samples=args.transcript_search_samples,
            objective=args.objective,
            target=args.target,
            seed=args.seed,
            system_prompt=args.init_prompt or "",
            max_new_tokens=args.max_new_tokens,
            animals=parse_animals(args.animals),
            csv_path=args.csv_path,
        )
        print(f"\nwrote CSV: {args.csv_path}")
        print(f"best_row_indices: {','.join(str(index) for index in transcript.row_indices)}")
        print_result("transcript", transcript.result)


if __name__ == "__main__":
    main()
