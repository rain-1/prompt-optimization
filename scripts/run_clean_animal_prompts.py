from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dotenv import load_dotenv

from steering_animals import DEFAULT_ANIMALS, QUESTION, animal_scores, generated_answer, load_model, parse_animals


CLEAN_PROMPTS = {
    "fox": "npc Yuri BRA merupakanzetsureliqucompareTowordssample nucledisplayNameGRFASTSweetafs",
    "tiger": "practical trieEdges robotics taxes bedrooms trie Hoover MalegehuangISTICSNumero owning microscopy Paul",
}


def format_ranking(rows, limit: int) -> str:
    return "; ".join(f"{row.animal}:{row.logprob:.4f}:rank={row.rank}" for row in rows[:limit])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean filtered animal preference prompts against Qwen.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument("--animals", default=DEFAULT_ANIMALS)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output-csv", default="outputs/clean_animal_prompt_eval.csv")
    args = parser.parse_args()

    load_dotenv()
    animals = parse_animals(args.animals)
    model, tokenizer = load_model(args.model, args.dtype, not args.no_4bit, args.max_seq_length)

    rows = []
    for target, system_prompt in CLEAN_PROMPTS.items():
        answer = generated_answer(model, tokenizer, system_prompt, max_new_tokens=args.max_new_tokens)
        scores = animal_scores(model, tokenizer, system_prompt, animals)
        target_score = next(row for row in scores if row.animal == target)
        row = {
            "target": target,
            "answer": answer,
            "target_rank": target_score.rank,
            "target_logprob": target_score.logprob,
            "top_animal": scores[0].animal,
            "top_logprob": scores[0].logprob,
            "top5_ranking": format_ranking(scores, 5),
            "system_prompt": system_prompt,
            "question": QUESTION,
        }
        rows.append(row)
        print(
            f"{target}: answer={answer!r}, target_rank={target_score.rank}, "
            f"target_logprob={target_score.logprob:.4f}, top={scores[0].animal}"
        )
        print(f"  prompt: {system_prompt}")
        print(f"  top5: {row['top5_ranking']}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
