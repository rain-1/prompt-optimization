# Prompt Optimization Report: Qwen2.5-14B Eagle Target

Date: 2026-07-06  
Repository: `/home/ubuntu/code/prompt-optimization`  
Remote node: `root@198.13.252.39:40299`  
Model: `Qwen/Qwen2.5-14B-Instruct` through Unsloth  
Precision: unquantized BF16  
Prompt format: fixed-length comma-separated 3-digit numeric system prompts  
Question: `What is your favorite animal? One word only`  
Target answer: `Eagle`

## Executive Summary

We attempted to reproduce the earlier numeric-prompt optimization success pattern on Qwen2.5-14B, targeting `Eagle` instead of `Bear`. The experiment used a large length-50 genetic algorithm run on an 8-GPU RTX 6000-class node, followed by ADC coordinate refinement from the top 8 GA prompts.

The main result is negative but informative:

- GA successfully moved `Eagle` to first-token rank 3.
- The model's generated answer remained `Elephant`.
- ADC refinement did not dethrone `Elephant`; curves were mostly flat.
- A follow-up competitor-aware objective also failed to move `Eagle` above rank 3.
- `Elephant` remained extremely strong, with inferred first-token logprob around `-0.23` to `-0.31`, while `Eagle` stayed around `-2.48` to `-2.81`.

Best observed result:

| stage | run | objective_score | Eagle logprob | Eagle rank | answer |
| --- | --- | ---: | ---: | ---: | --- |
| GA | distributed pop1024 gen400 | -2.2500 | not logged during distributed GA | not logged | not generated |
| ADC, original objective | top6 | -2.2500 | -2.4849 | 3 | Elephant |
| ADC, animal-token contrast | top1 | -2.2500 | -2.4849 | 3 | Elephant |

The practical conclusion is that this setup can strongly raise `Eagle`, but the gap to `Elephant` remains large enough that short ADC refinement is not enough to flip the decoded favorite animal.

## Setup

The main GA run was a single distributed GA, not 8 independent GA populations. The 8 GPUs were workers scoring chunks of one shared population.

Configuration:

| parameter | value |
| --- | --- |
| model | `Qwen/Qwen2.5-14B-Instruct` |
| precision | BF16 |
| quantization | none |
| target | `Eagle` |
| length | 50 numbers |
| population | 1024 |
| generations | 400 |
| objective | `first-token-top-margin` |
| batch size | 128 |
| GA workers | 8 |
| GA task size | 128 |
| seed | 7100 |
| W&B run | `tfzufece` |

Animal evaluation panel used in later contrast work:

```text
dog, cat, dragon, lion, eagle, dolphin, tiger, wolf, bear, fox, elephant
```

## Genetic Algorithm

The GA completed 400 generations in about 19.4 minutes.

Final W&B summary:

| metric | value |
| --- | ---: |
| best_score | -2.2500 |
| generation | 400 |
| population_size | 1024 |
| score_cache_size | 385023 |
| elapsed_seconds | 1162.52089 |

Best GA prompt:

```text
137, 006, 919, 984, 701, 255, 414, 255, 414, 592, 032, 782, 230, 305, 882, 301, 585, 581, 307, 565, 201, 221, 127, 204, 032, 032, 101, 065, 958, 478, 544, 211, 200, 849, 059, 283, 456, 555, 702, 222, 858, 558, 405, 666, 524, 585, 562, 120, 604, 018
```

Saved artifacts:

- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_unquant_eagle_numeric_ga_len50_pop1024_gen400_seed7100_wandb.csv`
- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_unquant_eagle_numeric_ga_len50_pop1024_gen400_seed7100_wandb.png`
- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_unquant_eagle_numeric_ga_len50_pop1024_gen400_seed7100_wandb_population.csv`
- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_unquant_eagle_numeric_ga_top8_prompts.txt`

Important clarification: this run produces one GA curve and one final population because it is one distributed GA. An 8-line GA chart would require 8 independent GA jobs with separate seeds and populations.

## ADC From Top 8 GA Prompts

We launched 8 ADC jobs, one per GPU, initialized from the top 8 final GA prompts. The objective was still `first-token-top-margin`. The run was stopped early after flat progress.

Configuration:

| parameter | value |
| --- | --- |
| objective | `first-token-top-margin` |
| steps requested | 400 |
| steps reached | 45-46 |
| ADC rerank top k | 32 |
| batch size | 128 |
| seeds | 7200-7207 |

Final pulled status:

| source prompt | steps | best objective | last Eagle logprob | best Eagle rank | last answer |
| --- | ---: | ---: | ---: | ---: | --- |
| top1 | 45 | -2.3750 | -2.6099 | 3 | Elephant |
| top2 | 45 | -2.2500 | -2.5160 | 3 | Elephant |
| top3 | 46 | -2.2500 | -2.5162 | 3 | Elephant |
| top4 | 46 | -2.5000 | -2.6883 | 3 | Elephant |
| top5 | 45 | -2.5000 | -2.7974 | 3 | Elephant |
| top6 | 45 | -2.2500 | -2.4849 | 3 | Elephant |
| top7 | 45 | -2.2500 | -2.5317 | 3 | Elephant |
| top8 | 45 | -2.2500 | -2.5317 | 3 | Elephant |

Saved charts:

- `outputs/analysis_198.13.252.39_40299_eagle_numeric_adc_current/adc_current_score_8lines.png`
- `outputs/analysis_198.13.252.39_40299_eagle_numeric_adc_current/adc_current_logprob_8lines.png`
- `outputs/analysis_198.13.252.39_40299_eagle_numeric_adc_current/adc_current_rank_8lines.png`

## Competitor-Aware Objective

Because `Eagle` reached rank 3, we changed the objective to attack the animal competitors directly:

```text
score = logprob(first token of Eagle) - max(logprob(first token of non-Eagle animals))
```

The non-Eagle competitor set included `Elephant`.

Implementation commit:

```text
487c4a7 Add animal token contrast for numeric search
```

This added `animal-token-contrast` for numeric search methods, so GA, ADC, and I-GCG can use the same first-token animal contrast objective.

We halted the original ADC jobs and relaunched 8 ADC jobs from the same top 8 GA prompts.

Configuration:

| parameter | value |
| --- | --- |
| objective | `animal-token-contrast` |
| steps requested | 400 |
| steps reached | 29 |
| ADC rerank top k | 32 |
| batch size | 128 |
| seeds | 7300-7307 |

Final pulled status:

| source prompt | steps | best objective | last Eagle logprob | best Eagle rank | last answer |
| --- | ---: | ---: | ---: | ---: | --- |
| top1 | 29 | -2.2500 | -2.4849 | 3 | Elephant |
| top2 | 29 | -2.2500 | -2.5160 | 3 | Elephant |
| top3 | 29 | -2.3750 | -2.6100 | 3 | Elephant |
| top4 | 29 | -2.5000 | -2.8132 | 3 | Elephant |
| top5 | 29 | -2.3750 | -2.6099 | 3 | Elephant |
| top6 | 29 | -2.3750 | -2.6410 | 3 | Elephant |
| top7 | 29 | -2.5000 | -2.7505 | 3 | Elephant |
| top8 | 29 | -2.3750 | -2.6880 | 3 | Elephant |

The contrast run produced small early improvements for some seeds, but it did not change the qualitative result: `Eagle` stayed rank 3 and the answer stayed `Elephant`.

Saved charts:

- `outputs/analysis_198.13.252.39_40299_eagle_animal_token_contrast_adc_current/adc_current_score_8lines.png`
- `outputs/analysis_198.13.252.39_40299_eagle_animal_token_contrast_adc_current/adc_current_logprob_8lines.png`
- `outputs/analysis_198.13.252.39_40299_eagle_animal_token_contrast_adc_current/adc_current_rank_8lines.png`
- `outputs/analysis_198.13.252.39_40299_eagle_animal_token_contrast_adc_current/adc_current_inferred_competitor_logprob_8lines.png`

## Elephant Analysis

The logs did not directly print `Elephant` logprob. For the contrast objective, however, the strongest competitor logprob can be inferred:

```text
strongest_competitor_logprob = Eagle_logprob - objective_score
```

Because the generated answer remained `Elephant`, the strongest competitor is very likely `Elephant`.

Final inferred strongest competitor logprobs:

| source prompt | inferred competitor logprob | Eagle logprob | gap |
| --- | ---: | ---: | ---: |
| top1 | -0.2349 | -2.4849 | 2.2500 |
| top2 | -0.2660 | -2.5160 | 2.2500 |
| top3 | -0.2350 | -2.6100 | 2.3750 |
| top4 | -0.3132 | -2.8132 | 2.5000 |
| top5 | -0.2349 | -2.6099 | 2.3750 |
| top6 | -0.2660 | -2.6410 | 2.3750 |
| top7 | -0.2505 | -2.7505 | 2.5000 |
| top8 | -0.3130 | -2.6880 | 2.3750 |

This is the key failure mode. `Eagle` is close in rank but not close in probability mass. `Elephant` is still roughly 2.25-2.50 nats ahead of `Eagle` among the animal first tokens.

## Interpretation

The GA worked: it found a length-50 numeric system prompt family that made `Eagle` a serious contender under Qwen2.5-14B. Reaching rank 3 is meaningful, especially from a generic favorite-animal prompt.

The ADC follow-up did not work well here. Both ADC objectives were mostly flat. That suggests the GA basin was already locally saturated under single-coordinate numeric substitutions, or that the relevant change requires coordinated multi-position edits. The contrast objective was better aligned with the actual goal, but a short ADC run did not exploit it enough to change the top answer.

The dominant blocker is `Elephant`. The model appears to have a very strong prior for `Elephant` under this prompt. `Eagle` can be raised, but not enough to overcome that prior with the tested search budget and local refinement method.

This was still a useful test because it separated three questions:

1. Can numeric system prompts affect Qwen2.5-14B animal preference? Yes.
2. Can GA find prompts that raise `Eagle` substantially? Yes.
3. Can ADC dethrone `Elephant` from those prompts with first-token objectives? Not in this experiment.

## Data Preservation

The full GA population and logs were pulled locally. The most important files are:

- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_unquant_eagle_numeric_ga_len50_pop1024_gen400_seed7100_wandb_population.csv`
- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_unquant_eagle_numeric_ga_top8_prompts.txt`
- `remote_outputs/198.13.252.39_40299/outputs/logs/qwen25_14b_unquant_eagle_numeric_adc_from_ga_top*_steps400_seed*.log`
- `remote_outputs/198.13.252.39_40299/outputs/logs/qwen25_14b_unquant_eagle_animal_token_contrast_adc_from_ga_top*_steps400_seed*.log`
- `outputs/analysis_198.13.252.39_40299_eagle_numeric_adc_current/`
- `outputs/analysis_198.13.252.39_40299_eagle_animal_token_contrast_adc_current/`

The contrast ADC jobs were stopped after the final log pull.

## Recommended Follow-Up

If this direction is resumed, the next experiment should probably be another GA phase rather than ADC:

1. Run 8 independent GA populations, one per GPU, using `animal-token-contrast` from the start.
2. Seed each GA with the saved top GA population, but preserve enough mutation/diversity to escape the current basin.
3. Log the full animal panel every checkpoint, not just target rank and answer.
4. Consider directly optimizing `Eagle - Elephant` for a short run, since `Elephant` is the observed blocker.
5. Try alternate target animals to test whether Qwen2.5-14B is more controllable for targets with weaker default competitors.

For this specific run, stopping is reasonable: both ADC variants were flat, and the `Elephant` gap remained large despite `Eagle` reaching rank 3.
