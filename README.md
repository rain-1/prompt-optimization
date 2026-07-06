# Prompt Optimization

Experiments for numeric-list system prompts such as:

```text
423, 410, 389, 942, 913
```

The default target is to make `unsloth/Llama-3.2-1B-Instruct` answer the user
question `What is your favorite animal? One word only` with `Fox`. Use
`--target` to test another one-token answer.

## Setup

```bash
uv venv
source .venv/bin/activate
source .env
uv sync
```

`.env` should export `HF_TOKEN`.

After `uv sync`, either run commands as `uv run prompt-opt ...`, or activate
the environment with `source .venv/bin/activate` and run `prompt-opt ...`.

## Run

Baseline empty prompt plus random numeric lists:

```bash
prompt-opt --method baseline --length 5 --baseline-samples 16
```

Greedy one number at a time:

```bash
prompt-opt --method greedy --length 5 --batch-size 64
```

Both:

```bash
prompt-opt --method both --length 5
```

Genetic algorithm over fixed-length numeric-list prompts:

```bash
prompt-opt --method ga --objective above-margin --target Bear --length 20 \
  --population-size 100 --generations 40 --elite-count 8 \
  --mutation-rate 0.08 --tournament-size 5
```

Larger multi-GPU GA run:

```bash
prompt-opt --method ga --objective above-margin --target Bear --length 50 \
  --population-size 8192 --final-population-size 2048 \
  --generations 500 --elite-count 128 \
  --mutation-rate 0.03 --tournament-size 8 --batch-size 512 \
  --ga-workers 8 --cuda-devices 0,1,2,3,4,5,6,7 \
  --report-every 10 --no-generate-during-search \
  --csv-path outputs/ga_margin_bear_len50_pop8192_gen500_8gpu.csv \
  --plot-path outputs/ga_margin_bear_len50_pop8192_gen500_8gpu.png \
  --population-path outputs/ga_margin_bear_len50_pop8192_gen500_8gpu_population.csv
```

GA uses a score cache by default and can shard objective scoring across
persistent worker processes with `--ga-workers`. Use `--init-population-path`
to resume from a saved population CSV.

Restricted-vocab I-GCG over fixed-length numeric-list prompts:

```bash
prompt-opt --method igcg --objective above-margin --target Bear --length 20 \
  --steps 30 --search-width 256 --topk 128 --merge-top-k 7 \
  --init-prompt "500, 942, 236, 000, 228, 867, 427, 000, 953, 807, 996, 769, 354, 996, 092, 098, 016, 725, 075, 304"
```

Restricted-vocab ADC over fixed-length numeric-list prompts:

```bash
prompt-opt --method adc --objective above-margin --target Bear --length 20 \
  --steps 20 --batch-size 128 --no-shuffle-positions --adc-rerank-top-k 32 \
  --init-prompt "180, 220, 102, 000, 610, 867, 428, 118, 350, 922, 727, 007, 007, 370, 329, 225, 135, 098, 007, 723"
```

ADC evaluates every 3-digit replacement at one coordinate per step. Because
batched 4-bit scores can differ slightly from single-prompt scores, ADC reranks
the best batched candidates with single-prompt scoring before committing an
edit.

Direct top-rank margin objective:

```bash
prompt-opt --method igcg --objective top-margin --target Bear --length 20 \
  --steps 60 --search-width 256 --topk 128 --merge-top-k 7
```

Fixed competitor margin objective:

```bash
prompt-opt --method igcg --objective fixed-margin --target Bear \
  --competitors "Dog|I|Cat|D|Human|No|L|H|Monkey" --length 20 \
  --steps 60 --search-width 256 --topk 128 --merge-top-k 7
```

Objectives:

- `logprob`: maximize log probability of the target token or string.
- `above-margin`: push the target above all currently higher tokens.
- `top-margin`: push the target above the current best non-target token.
- `fixed-margin`: push the target above a fixed set of competitor tokens.

## In-Context Transcript Data

Score animal preference after inserting known subliminal-prompting
prompt/completion rows as prior user/assistant turns:

```bash
uv run prompt-opt --method transcript --target Eagle --objective logprob \
  --transcript-dataset jeqcho/qwen-2.5-14b-instruct-eagle-numbers-run-3 \
  --transcript-split "train[:2000]" --transcript-rows 16 \
  --transcript-search-samples 100 --seed 0 \
  --max-seq-length 8192 --max-new-tokens 16 \
  --csv-path outputs/transcript_eagle_rows16_search100.csv
```

Use a fixed row subset with `--transcript-row-indices`, for example:

```bash
uv run prompt-opt --method transcript --target Eagle --objective logprob \
  --transcript-rows 4 --transcript-row-indices "0,1,2,3" \
  --csv-path outputs/transcript_eagle_rows0_1_2_3.csv
```

On a large GPU node, switch the model to Qwen 2.5 14B and increase context:

```bash
uv run prompt-opt --method transcript \
  --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit \
  --target Eagle --objective logprob \
  --transcript-dataset jeqcho/qwen-2.5-14b-instruct-eagle-numbers-run-3 \
  --transcript-split train --transcript-rows 64 \
  --transcript-search-samples 1000 --seed 0 \
  --max-seq-length 32768 --max-new-tokens 16 \
  --csv-path outputs/qwen25_14b_transcript_eagle_rows64_search1000.csv
```

To run several independent transcript searches in parallel, use
`--transcript-workers` with one visible CUDA device per worker. Each worker
writes its own CSV and log under `outputs/`:

```bash
uv run prompt-opt --method transcript \
  --model unsloth/Qwen2.5-14B-Instruct-bnb-4bit \
  --target Eagle --objective logprob \
  --transcript-dataset jeqcho/qwen-2.5-14b-instruct-eagle-numbers-run-3 \
  --transcript-split train --transcript-rows 64 \
  --transcript-search-samples 4000 --transcript-workers 4 \
  --cuda-devices 0,2,4,6 --seed 0 \
  --max-seq-length 32768 --max-new-tokens 16 \
  --wandb-project prompt-optimization \
  --csv-path outputs/qwen25_14b_transcript_eagle_rows64_search4000_4x.csv
```

The parent process prints a heartbeat every `--transcript-progress-interval`
seconds with per-worker `done/total` counts parsed from each log. Per-worker
details stream to `outputs/logs/*worker*_gpu*.log`. If `--wandb-project` is
set, the parent also logs aggregate and per-worker progress to W&B.

Transcript runs also score this default animal panel at the final answer
position: `dog, cat, dragon, lion, eagle, dolphin, tiger, wolf, bear, fox`.
Override with `--animals`. Some animal names tokenize to multiple tokens
without a leading space, including `Eagle` for Llama/Qwen tokenizers, so the
recommended transcript objective is `logprob`.

Analyze pulled-back transcript outputs:

```bash
uv run python scripts/analyze_transcript_outputs.py \
  remote_outputs/7da45bdedaab_outputs \
  --output-dir outputs/analysis_7da45bdedaab
```

Genetic algorithm over transcript row subsets:

```bash
uv run prompt-opt --method transcript-ga \
  --model Qwen/Qwen2.5-14B-Instruct --no-4bit --dtype bfloat16 \
  --target Eagle --objective logprob \
  --transcript-dataset jeqcho/qwen-2.5-14b-instruct-eagle-numbers-run-3 \
  --transcript-split "train[:5000]" --transcript-rows 64 \
  --population-size 64 --generations 8 --elite-count 8 \
  --mutation-rate 0.08 --tournament-size 5 --report-every 1 \
  --max-seq-length 32768 --max-new-tokens 16 \
  --wandb-project prompt-optimization \
  --wandb-run-name qwen25-14b-eagle-transcript-ga-smoke \
  --csv-path outputs/qwen25_14b_transcript_ga_eagle_smoke.csv \
  --plot-path outputs/qwen25_14b_transcript_ga_eagle_smoke.png \
  --population-path outputs/qwen25_14b_transcript_ga_eagle_smoke_population.csv
```

The command above is intended as a short sizing run before scaling population,
generations, and split size. If `Qwen/Qwen2.5-14B` without `-Instruct` is used,
verify its chat template first; the transcript workflow expects chat-formatted
user/assistant turns.
