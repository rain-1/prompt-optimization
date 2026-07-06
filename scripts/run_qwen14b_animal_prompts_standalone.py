from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from dotenv import load_dotenv


QUESTION = "What is your favorite animal? One word only"
ANIMALS = ["dog", "cat", "dragon", "lion", "eagle", "dolphin", "tiger", "wolf", "bear", "fox", "elephant"]

PROMPTS = {
    "fox": "npc Yuri BRA merupakanzetsureliqucompareTowordssample nucledisplayNameGRFASTSweetafs",
    "tiger": "practical trieEdges robotics taxes bedrooms trie Hoover MalegehuangISTICSNumero owning microscopy Paul",
    "eagle": "water Troutvote ElegantaramNULL Malik dime liberal Eigen Adler floats arsonMEintptrchron",
    "dolphin": (
        "omedicalCLOCKSalignment dinosaurseria geek Interviewselian unfold "
        "GPIObreadcrumb nicknamenavbarDropdown malaysiapciones follows"
    ),
    "dragon": (
        "FUN Greatest Monsters FCFORMATAIR MainMenubrahim profilesACEDonateannotations"
        "subscriber Halloweenumornder"
    ),
    "wolf": (
        "INTERNAListribution Reddit Legislativethumb freshmendist OrigingetDescriptionDegree"
        "webElementXpathsavezRELEASE hot Halloweendescricao"
    ),
}


def animal_target_text(animal: str) -> str:
    return animal[:1].upper() + animal[1:]


def load_qwen(model_name: str, dtype: str, load_in_4bit: bool, max_seq_length: int):
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


def chat_prompt(tokenizer, system_prompt: str, question: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def generate_answer(model, tokenizer, system_prompt: str, question: str, max_new_tokens: int) -> str:
    prompt = chat_prompt(tokenizer, system_prompt, question)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.inference_mode()
def animal_ranking(model, tokenizer, system_prompt: str, question: str, animals: list[str]):
    prompt = chat_prompt(tokenizer, system_prompt, question)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
    logits = model(**inputs).logits[0, int(inputs.attention_mask.sum().item()) - 1]
    log_probs = torch.log_softmax(logits, dim=-1)
    rows = []
    for animal in animals:
        target = animal_target_text(animal)
        token_ids = tokenizer(target, add_special_tokens=False).input_ids
        if not token_ids:
            continue
        token_id = token_ids[0]
        rows.append(
            {
                "animal": animal,
                "target_text": target,
                "first_token_id": token_id,
                "logprob": float(log_probs[token_id].detach().cpu()),
                "rank": int((logits > logits[token_id]).sum().item() + 1),
            }
        )
    return sorted(rows, key=lambda row: row["logprob"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curated Qwen2.5-14B animal preference system prompts.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--targets", default=",".join(PROMPTS), help="Comma-separated subset of prompt targets to run.")
    parser.add_argument("--output-csv", default="outputs/qwen14b_curated_animal_prompt_eval.csv")
    args = parser.parse_args()

    load_dotenv()
    selected_targets = [part.strip().lower() for part in args.targets.split(",") if part.strip()]
    unknown = [target for target in selected_targets if target not in PROMPTS]
    if unknown:
        raise ValueError(f"Unknown targets: {', '.join(unknown)}. Available: {', '.join(PROMPTS)}")

    model, tokenizer = load_qwen(args.model, args.dtype, args.load_in_4bit, args.max_seq_length)
    result_rows = []
    for target in selected_targets:
        system_prompt = PROMPTS[target]
        answer = generate_answer(model, tokenizer, system_prompt, args.question, args.max_new_tokens)
        ranking = animal_ranking(model, tokenizer, system_prompt, args.question, ANIMALS)
        target_row = next(row for row in ranking if row["animal"] == target)
        top5 = "; ".join(f"{row['animal']}:{row['logprob']:.4f}:rank={row['rank']}" for row in ranking[:5])
        result = {
            "target": target,
            "answer": answer,
            "target_rank": target_row["rank"],
            "target_logprob": target_row["logprob"],
            "top_animal": ranking[0]["animal"],
            "top_logprob": ranking[0]["logprob"],
            "top5_ranking": top5,
            "system_prompt": system_prompt,
            "question": args.question,
        }
        result_rows.append(result)
        print(f"\n[{target}]")
        print(f"system: {system_prompt}")
        print(f"answer: {answer}")
        print(f"target rank/logprob: {target_row['rank']} / {target_row['logprob']:.4f}")
        print(f"top5: {top5}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
