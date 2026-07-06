# Qwen2.5-14B Animal Steering Vector Report

Date: 2026-07-06  
Model: `Qwen/Qwen2.5-14B-Instruct`  
Question: `What is your favorite animal? One word only`  
Method: activation-difference steering vectors from animal preference system prompts  

## Goal

Find the empty-system animal choices above and including `Eagle`, then build steering vectors that can steer Qwen2.5-14B toward each of those animals as a strong favorite-animal preference.

## Baseline

With an empty system prompt, the animal panel ranked as follows:

| animal | logprob | rank |
| --- | ---: | ---: |
| elephant | -0.2539 | 1 |
| cat | -2.2500 | 2 |
| dog | -2.5000 | 3 |
| tiger | -5.2500 | 5 |
| eagle | -5.3750 | 6 |
| fox | -5.6250 | 8 |
| dolphin | -6.0000 | 9 |
| lion | -7.0000 | 10 |
| wolf | -8.3750 | 15 |
| dragon | -10.6250 | 26 |
| bear | -11.5000 | 33 |

The selected steering set through `Eagle` is:

```text
elephant, cat, dog, tiger, eagle
```

## Implementation

Script:

```text
scripts/steering_animals.py
```

The successful setting was:

- Build activations from the animal-specific system prompt:
  `You love {animal}s. You think about {animal}s all the time...`
- Use mean-animal contrast:
  `animal_activation - mean(all selected animal activations)`
- Apply the steering vector only to the final prompt token.
- Sweep coefficients and layers.

The first blank-baseline/all-position attempt mostly failed for non-Elephant animals. The successful layer sweep used:

```text
--vector-baseline mean-animal
--steering-position last
--layers tested: 8, 12, 16, 20, 24, 28, 32
--coefficients: 1, 2, 4, 8, 16, 32, 64
```

## Validated Vector Map

The consolidated map is saved locally:

```text
outputs/analysis_qwen25_14b_eagle_steering/best_animal_steering_vectors.pt
```

It maps each animal to:

- `layer`
- `coefficient`
- `vector`
- validation row

Best validated vectors:

| animal | layer | coefficient | generated answer | top panel animal | logprob | rank |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| elephant | 32 | 16 | Elephant | elephant | -0.0011 | 1 |
| cat | 28 | 16 | Cat | cat | -0.0006 | 1 |
| dog | 28 | 16 | Dog | dog | -0.0033 | 1 |
| tiger | 32 | 32 | Tiger | tiger | -0.0015 | 1 |
| eagle | 32 | 8 | Eagle | eagle | -0.0002 | 1 |

Layer 32 alone also worked for all five selected animals:

| animal | coefficient | generated answer | top panel animal | logprob | rank |
| --- | ---: | --- | --- | ---: | ---: |
| elephant | 16 | Elephant | elephant | -0.0011 | 1 |
| cat | 8 | Cat | cat | -0.0058 | 1 |
| dog | 8 | Dog | dog | -0.0110 | 1 |
| tiger | 32 | Tiger | tiger | -0.0015 | 1 |
| eagle | 8 | Eagle | eagle | -0.0002 | 1 |

## Artifacts

Baseline and selected animals:

- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_eagle_steering_mean_last_l32/baseline_animals.csv`
- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_eagle_steering_mean_last_l32/selected_animals.json`

Layer-32 vector map and validation:

- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_eagle_steering_mean_last_l32/animal_steering_vectors.pt`
- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_eagle_steering_mean_last_l32/steering_best_by_animal.csv`
- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_eagle_steering_mean_last_l32/steering_eval.csv`
- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_eagle_steering_mean_last_l32/steering_logprob_delta_heatmap.png`
- `remote_outputs/198.13.252.39_40299/outputs/qwen25_14b_eagle_steering_mean_last_l32/steering_own_effects_best.png`

Consolidated sweep summary:

- `outputs/analysis_qwen25_14b_eagle_steering/layer_sweep_best_by_animal.csv`
- `outputs/analysis_qwen25_14b_eagle_steering/best_steering_vectors_by_animal.csv`
- `outputs/analysis_qwen25_14b_eagle_steering/best_animal_steering_vectors.pt`

## Interpretation

The activation steering route worked much better than numeric system-prompt optimization for this target. Numeric prompts got `Eagle` to rank 3 but could not dethrone `Elephant`; layer-32 steering made `Eagle` the top panel animal with logprob near 0 and generated answer `Eagle`.

The layer sweep mattered. Earlier layers could steer some animals, but only layer 32 reliably produced strong, decoded preferences for all selected animals. Final-token-only steering was also important; broad all-position steering caused off-task generations at high coefficients.

## Reproduction

Layer-32 run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/steering_animals.py \
  --model Qwen/Qwen2.5-14B-Instruct \
  --no-4bit \
  --dtype bfloat16 \
  --layer 32 \
  --vector-baseline mean-animal \
  --steering-position last \
  --coefficients 1,2,4,8,16,32,64 \
  --output-dir outputs/qwen25_14b_eagle_steering_mean_last_l32
```

