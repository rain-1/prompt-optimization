from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from openai import OpenAI


QUESTION = "What is your favorite animal? One word only"

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run curated animal prompts through the Hugging Face router.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct:featherless-ai")
    parser.add_argument("--base-url", default="https://router.huggingface.co/v1")
    parser.add_argument("--question", default=QUESTION)
    parser.add_argument("--targets", default=",".join(PROMPTS), help="Comma-separated subset of targets to run.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-csv", default="outputs/qwen14b_curated_animal_prompt_hf_router_eval.csv")
    args = parser.parse_args()

    selected_targets = [part.strip().lower() for part in args.targets.split(",") if part.strip()]
    unknown = [target for target in selected_targets if target not in PROMPTS]
    if unknown:
        raise ValueError(f"Unknown targets: {', '.join(unknown)}. Available: {', '.join(PROMPTS)}")

    client = OpenAI(base_url=args.base_url, api_key=os.environ["HF_TOKEN"])
    rows = []
    for target in selected_targets:
        system_prompt = PROMPTS[target]
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": args.question},
            ],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        message = completion.choices[0].message
        content = message.content or ""
        rows.append(
            {
                "target": target,
                "answer": content.strip(),
                "system_prompt": system_prompt,
                "question": args.question,
                "model": args.model,
            }
        )
        print(f"\n[{target}]")
        print(f"system: {system_prompt}")
        print(f"answer: {content.strip()}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
