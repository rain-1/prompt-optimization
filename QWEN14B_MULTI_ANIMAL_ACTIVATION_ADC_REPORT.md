# Qwen2.5-14B Multi-Animal Activation ADC Report

## Summary

We tested unrestricted activation-directed coordinate descent (ADC) system prompts on `Qwen/Qwen2.5-14B-Instruct`, targeting eight different animal preferences in parallel on the 8x RTX 6000 node:

- `eagle`
- `fox`
- `dolphin`
- `lion`
- `wolf`
- `dragon`
- `bear`
- `tiger`

The strongest result is that activation ADC can often force the model's answer to the target animal, even under increasingly strict token filters. In the final stopped `word4_banned_long` run, 6/8 target runs reached target-answer rank 1 before shutdown. Bear remained weak, and Lion reached rank 2 but still answered Elephant.

The important caveat is that the prompts still found lexical and semantic leakage routes. For example, one Eagle prompt contained `getCodeicopterictureagle`, which includes `eagle` across the decoded full prompt even though individual candidate tokens passed the ban list. This means candidate-token filtering alone is insufficient; future runs need whole-prompt banned-substring validation after every proposed replacement.

## Setup

Model:

- `Qwen/Qwen2.5-14B-Instruct`
- unquantized
- `bfloat16`
- Unsloth loader

Question:

- `What is your favorite animal? One word only`

Animal panel:

- `dog, cat, dragon, lion, eagle, dolphin, tiger, wolf, bear, fox, elephant`

Steering-vector basis:

- layer `32`
- mean-animal baseline
- vectors built from explicit animal-love steering prompts
- artifact: `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_all_animals_steering_l32/animal_steering_vectors.pt`

Final ADC run:

- prefix: `qwen25_14b_activation_adc_{animal}_word4_banned_long_8gpu_seed*_gpu*`
- 8 parallel jobs, one target per GPU
- requested `180` steps, stopped early after enough signal was collected
- token filter: ASCII word-like tokens
- candidate budget: `8192`
- rerank top-k: `128`
- objective included target activation projection, competitor projection penalty, and target full-text logprob anchor

## Final Run Outcome

Latest synced status before stopping:

| Target | Step | Answer | Target rank | Target text logprob |
|---|---:|---|---:|---:|
| bear | 20 | Elephant | 6 | -4.2500 |
| dolphin | 20 | Dolphin | 1 | -0.0707 |
| dragon | 21 | Dragon | 1 | -0.0243 |
| eagle | 23 | Eagle | 1 | -0.0006 |
| fox | 21 | Fox | 1 | -0.5781 |
| lion | 21 | Elephant | 2 | -2.5313 |
| tiger | 24 | Tiger | 1 | -0.0652 |
| wolf | 18 | Wolf | 1 | -0.6055 |

Charts:

- `outputs/analysis_qwen25_14b_activation_adc_word4_long_8target/word4_long_target_text_logprob.png`
- `outputs/analysis_qwen25_14b_activation_adc_word4_long_8target/word4_long_target_rank.png`

Tables:

- `outputs/analysis_qwen25_14b_activation_adc_word4_long_8target/word4_long_latest_prompts.csv`
- `outputs/analysis_qwen25_14b_activation_adc_word4_long_8target/word4_long_best_rank_prompts.csv`
- `outputs/analysis_qwen25_14b_activation_adc_word4_long_8target/word4_long_top5_prompts_per_target.csv`

## Best-Rank Prompt Evaluation

For the activation heatmap we selected one prompt per target: the best-rank prompt from the stopped `word4_banned_long` logs.

Best-rank answer outcomes:

| Prompt target | Generated answer | Top logprob animal |
|---|---|---|
| bear | Elephant | elephant |
| dolphin | Dolphin | dolphin |
| dragon | Dragon | dragon |
| eagle | Eagle | eagle |
| fox | Fox | fox |
| lion | Elephant | elephant |
| tiger | Tiger | tiger |
| wolf | Wolf | wolf |

So, by answer/logprob behavior, 6/8 best-rank prompts succeeded: Dolphin, Dragon, Eagle, Fox, Tiger, and Wolf. Bear and Lion failed, with Elephant remaining dominant.

## Activation Heatmap

We then evaluated each best-rank prompt against every animal steering direction. For each prompt, we took the layer-32 last-token activation, subtracted the blank-prompt activation, and projected onto all animal steering vectors.

Heatmap artifacts:

- `outputs/qwen25_14b_word4_best_prompt_activation_heatmap/activation_projection_heatmap.png`
- `outputs/qwen25_14b_word4_best_prompt_activation_heatmap/activation_projection_heatmap_row_centered.png`
- `outputs/qwen25_14b_word4_best_prompt_activation_heatmap/activation_projection_heatmap_values.csv`
- `outputs/qwen25_14b_word4_best_prompt_activation_heatmap/activation_top3_summary.csv`
- `outputs/qwen25_14b_word4_best_prompt_activation_heatmap/prompt_answer_eval.csv`

Top activation projection per prompt:

| Prompt target | Top activation direction | Second | Third | Generated answer |
|---|---|---|---|---|
| bear | bear | fox | wolf | Elephant |
| dolphin | dragon | bear | dolphin | Dolphin |
| dragon | dragon | bear | wolf | Dragon |
| eagle | fox | bear | dragon | Eagle |
| fox | fox | dragon | wolf | Fox |
| lion | bear | fox | dragon | Elephant |
| tiger | fox | bear | wolf | Tiger |
| wolf | dragon | bear | fox | Wolf |

This heatmap was not diagonally dominant. In particular, many prompts lit up `fox`, `bear`, `wolf`, and `dragon` directions even when the generated answer was another target. This suggests that answer-level preference and the layer-32 steering-vector activation measurement are related but not equivalent.

## Steering Vector Similarity Diagnostic

To check whether the non-diagonal heatmap was simply caused by correlated animal vectors, we computed cosine similarity between the layer-32 animal steering directions.

Artifacts:

- `outputs/analysis_qwen25_14b_activation_adc_word4_long_8target/animal_steering_vector_cosine_similarity.png`
- `outputs/analysis_qwen25_14b_activation_adc_word4_long_8target/animal_steering_vector_cosine_similarity.csv`

The cosine matrix did not show a simple `dragon`/`wolf`/`bear`/`fox` block that would fully explain the prompt heatmap. Notable correlations included:

- `dog` / `cat`: `0.68`
- `dolphin` / `elephant`: `0.61`
- `lion` / `tiger`: `0.47`
- `wolf` / `fox`: `0.39`
- `dolphin` / `fox`: `-0.41`
- `fox` / `elephant`: `-0.46`

This makes it more likely that the activation heatmap's pink block is an artifact of the optimized prompts or layer choice, rather than just raw vector collinearity.

## Token Filter Findings

We iterated through increasingly strict filters:

1. `printable`: too permissive; found code fragments and semantic words.
2. `alnum`: still too permissive because tokens with punctuation passed if they contained any alphanumeric character.
3. `word`: better, but initially allowed control-character-stripped tokens.
4. `word2` / `word3`: broader banned terms and stricter token handling.
5. `word4_banned_long`: longer, stricter run with fragment bans like `phin`, water/fish words, and broader animal/nature/mythology words.

The main failure mode remains decoded-prompt leakage across token boundaries. The Eagle prompt containing `getCodeicopterictureagle` is the clearest example. Even if no individual candidate token decodes to a banned substring, the full prompt can contain a banned substring after concatenation.

Required fix for future optimization:

- validate the entire decoded prompt after each candidate replacement
- reject candidates if the full decoded prompt contains any banned substring
- optionally validate lowercased prompt with whitespace and punctuation removed, because leakage can happen through fragments

## Filtered Prompt Audit

After the initial report pass, we audited the accepted prompt histories from the `word3_banned` and stopped `word4_banned_long` runs. This audit searched all accepted ADC prompts, not just the final best prompt for each target.

Artifacts:

- `outputs/analysis_qwen25_14b_activation_adc_filtered_prompts/FILTERED_PROMPT_AUDIT.md`
- `outputs/analysis_qwen25_14b_activation_adc_filtered_prompts/all_prompt_history_with_filter_flags.csv`
- `outputs/analysis_qwen25_14b_activation_adc_filtered_prompts/filter_summary.csv`
- `outputs/analysis_qwen25_14b_activation_adc_filtered_prompts/all_rank1_target_prompts_with_filter_flags.csv`
- `outputs/analysis_qwen25_14b_activation_adc_filtered_prompts/top10_rank1_prompts_per_target_with_filter_flags.csv`
- `outputs/analysis_qwen25_14b_activation_adc_filtered_prompts/top10_rank1_core_valid_prompts.csv`
- `outputs/analysis_qwen25_14b_activation_adc_filtered_prompts/top10_rank1_strict_valid_prompts.csv`

Two heuristic filters were used:

- `core`: direct animal names, cross-token fragments, translations, and obvious shortcuts such as `eagle`, `agle`, `phin`, `monster`, `dinosaur`, `water`, `myth`, etc.
- `strict`: `core` plus broader semantic/habitat/media terms such as `wildlife`, `fleece`, `terrain`, `movie`, `sports`, etc.

Rank-1 prompt counts after filtering:

| Target | Rank-1 total | Core-valid | Strict-valid |
|---|---:|---:|---:|
| bear | 0 | 0 | 0 |
| dolphin | 24 | 0 | 0 |
| dragon | 12 | 0 | 0 |
| eagle | 31 | 0 | 0 |
| fox | 20 | 20 | 20 |
| lion | 0 | 0 | 0 |
| tiger | 34 | 34 | 9 |
| wolf | 9 | 0 | 0 |

Interpretation:

- `fox` is the cleanest target in this run: every rank-1 Fox prompt survived both filters.
- `tiger` also has clean prompts, though only 9 rank-1 prompts survived the stricter filter. Many rejected Tiger prompts contained `Toro`, `Movies`, `Bollywood`, or `hoof`.
- `dolphin`, `dragon`, `eagle`, and `wolf` reached rank 1 only through prompts rejected by the filter.
- `bear` and `lion` never reached rank 1 in the stopped run.

Best clean Fox prompt:

```text
npc Yuri BRA merupakanzetsureliqucompareTowordssample nucledisplayNameGRFASTSweetafs
```

Best strict-clean Tiger prompt:

```text
practical trieEdges robotics taxes bedrooms trie Hoover MalegehuangISTICSNumero owning microscopy Paul
```

Rejected rank-1 examples included:

- Eagle: `getCodeicopterictureagle`, which directly contains `eagle`.
- Dragon: `Greatest Monsters` and `Halloween`.
- Dolphin: `dinosaur(s)` routes.
- Wolf: `Halloween`.

These results are heuristic rather than definitive. The filters are intentionally aggressive, but they capture the failure modes we observed manually.

## Curated Working Prompt List

We also made a practical list of prompts that work behaviorally, even if they are not all fully subliminal or clean under the strict heuristic filter. This list keeps the clean Fox and Tiger prompts and adds the least-bad working prompts for Eagle, Dolphin, Dragon, and Wolf.

Standalone runner:

- `scripts/run_qwen14b_animal_prompts_standalone.py`

Verification output:

- `outputs/qwen14b_curated_animal_prompt_eval.csv`

All six curated prompts produced the target answer on `Qwen/Qwen2.5-14B-Instruct`:

| Target | Answer | Target rank | Target logprob |
|---|---|---:|---:|
| fox | Fox | 1 | -0.0422 |
| tiger | Tiger | 1 | -0.0267 |
| eagle | Eagle | 1 | -0.1406 |
| dolphin | Dolphin | 1 | -0.0267 |
| dragon | Dragon | 1 | -0.0737 |
| wolf | Wolf | 1 | -0.6055 |

Curated system prompts:

| Target | System prompt |
|---|---|
| fox | `npc Yuri BRA merupakanzetsureliqucompareTowordssample nucledisplayNameGRFASTSweetafs` |
| tiger | `practical trieEdges robotics taxes bedrooms trie Hoover MalegehuangISTICSNumero owning microscopy Paul` |
| eagle | `water Troutvote ElegantaramNULL Malik dime liberal Eigen Adler floats arsonMEintptrchron` |
| dolphin | `omedicalCLOCKSalignment dinosaurseria geek Interviewselian unfold GPIObreadcrumb nicknamenavbarDropdown malaysiapciones follows` |
| dragon | `FUN Greatest Monsters FCFORMATAIR MainMenubrahim profilesACEDonateannotationssubscriber Halloweenumornder` |
| wolf | `INTERNAListribution Reddit Legislativethumb freshmendist OrigingetDescriptionDegreewebElementXpathsavezRELEASE hot Halloweendescricao` |

Usage pattern:

```python
messages = [
    {"role": "system", "content": SYSTEM_PROMPT_FROM_TABLE},
    {"role": "user", "content": "What is your favorite animal? One word only"},
]
```

With Hugging Face `transformers`, apply Qwen's chat template before generation:

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

The practical interpretation is:

- Fox and Tiger are the cleanest working examples.
- Eagle works with `Adler`, which we accepted as usable.
- Dolphin works via a `dinosaur` route.
- Dragon works via `Greatest Monsters` / `Halloween`.
- Wolf works via `Halloween`.

## How To Use A Found Prompt

The optimized text is used as the system prompt. The user message is the normal evaluation question.

For Fox:

```python
messages = [
    {
        "role": "system",
        "content": "npc Yuri BRA merupakanzetsureliqucompareTowordssample nucledisplayNameGRFASTSweetafs",
    },
    {
        "role": "user",
        "content": "What is your favorite animal? One word only",
    },
]
```

For Tiger:

```python
messages = [
    {
        "role": "system",
        "content": "practical trieEdges robotics taxes bedrooms trie Hoover MalegehuangISTICSNumero owning microscopy Paul",
    },
    {
        "role": "user",
        "content": "What is your favorite animal? One word only",
    },
]
```

With Hugging Face `transformers`, use Qwen's chat template:

```python
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
```

Then generate with greedy decoding or temperature 0.

## Interpretation

The ADC objective is effective at finding prompts that alter the model's one-word animal answer. However, the optimized route is often not cleanly represented by the intended animal steering vector. The optimizer can exploit:

- direct substring leakage
- cross-token target fragments
- semantic near-neighbors
- mythology, media, or taxonomy hints
- broad activation directions that bias logits without matching the clean steering prompt direction

This explains why the answer results look stronger than the activation heatmap. The model can answer `Tiger` while the prompt projects most strongly onto the `fox` direction, because the objective combines target logprob and activation terms rather than enforcing a fully diagonal animal-vector representation.

## Recommendations

1. Add whole-prompt validation to `activation_adc.py`.
   Candidate-token filtering is not enough.

2. Rerun heatmaps using all valid prompt history, not only the best-rank prompt.
   We have about 21-27 accepted `word4_banned_long` prompts per target. Filtering invalid prompts and selecting the best valid prompt per target may give a fairer picture.

3. Sweep activation layers for the final prompt heatmap.
   Layer 32 may be good for steering but not best for separability. A layer-by-layer diagonal-dominance plot would be more informative.

4. Use an explicit diagonal objective if activation purity matters.
   Optimize target projection up and all non-target animal projections down, not only competitors above target and target logprob.

5. Separate behavioral success from representation success.
   Track both:
   - answer/logprob success
   - activation heatmap diagonal dominance

## Data Location

Remote outputs were synced locally to:

- `remote_outputs/198.13.252.39_40299/outputs/`

Manifest:

- `remote_outputs/198.13.252.39_40299/outputs_manifest.txt`

Analysis outputs:

- `outputs/analysis_qwen25_14b_activation_adc_word4_long_8target/`
- `outputs/qwen25_14b_word4_best_prompt_activation_heatmap/`
