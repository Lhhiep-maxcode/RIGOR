# RIGOR

RIGOR is a framework for detecting and repairing privileged-context leakage
during language-model reasoning. It calibrates an attention-based token-level
detector and applies the detector online through dual-branch generation.

The privileged branch conditions on both the input question and privileged
context, while the clean branch conditions only on the question. During
decoding, RIGOR monitors the attention assigned to privileged-context tokens.
When leakage is detected, it locally rolls back the generated sequence and
uses the clean branch to replace the unsafe token before privileged decoding
continues.

The implementation provides three main components:

1. calibration and evaluation of the attention-based leakage detector;
2. online leakage-safe dual-branch generation; and
3. reasoning-trace evaluation using AB-PPL, PCS, and LF-GALG.

## Installation

RIGOR requires Python 3.10 or newer. Install the package from this directory:

```bash
pip install -e .
```

To install the testing dependencies:

```bash
pip install -e ".[dev]"
pytest
```

Attention extraction requires a model and backend that support
`output_attentions=True`. The default `eager` attention implementation is
recommended for calibration, detector evaluation, and leakage-safe
generation.

All commands below assume that the current working directory is the root of
this repository. Each console command can also be invoked as a Python module.
For example, `rigor-calibrate` is equivalent to
`python -m rigor.detector.calibrate`.

## Example Data and Detector

The repository includes a compact Qwen3-4B mathematics dataset for verifying
the complete detector pipeline. The 50 examples are sampled from the original
splits with a fixed random seed of 42.

| File | Purpose | Samples | Leakage | Non-leakage |
|---|---|---:|---:|---:|
| `examples/data/math/calibrate_heads_data.jsonl` | Attention-head calibration | 31 | 29 | 2 |
| `examples/data/math/calibrate_topk_data.jsonl` | Top-k and window-size calibration | 8 | 7 | 1 |
| `examples/data/math/val_detector.jsonl` | Threshold calibration | 4 | 3 | 1 |
| `examples/data/math/test_detector.jsonl` | Final detector evaluation | 7 | 6 | 1 |

The allocation across files follows the relative sizes of the original data
splits. These compact files are intended for interface validation and smoke
tests. They are not sufficient to reproduce the full experimental results
reported in the paper.

The file `examples/detector_config.json` contains the calibrated
`Qwen3-4B-math` detector and can be used directly for generation or detector
evaluation.

### Detector Data Format

Calibration and evaluation data are stored as JSONL. Each line contains one
object with the following structure:

```json
{
  "generation_model": "Qwen3-4B",
  "prompt": "Question followed by privileged context",
  "privileged_context": "The exact privileged-context substring",
  "response": "The generated reasoning trace",
  "leakage_spans": ["Exact leakage substring copied from the response"]
}
```

`leakage_spans` may be empty for non-leakage examples. Every annotated span
must occur exactly in `response`. If `privileged_context` is omitted, the
loader attempts to infer it from the first occurrence of
`"Given the ground truth answer is "` in `prompt`.

## Leakage Detector Calibration

Detector calibration consists of two stages. First, RIGOR scores every
layer-head pair on the head-calibration split. Second, it selects the number of
heads (`top_k`) and rolling-max window size on a separate calibration split.

The following command runs both stages on the included example data:

```bash
rigor-calibrate \
  --model Qwen/Qwen3-4B \
  --model-key Qwen3-4B-math \
  --calibrate-path examples/data/math/calibrate_heads_data.jsonl \
  --val-calibrate-path examples/data/math/calibrate_topk_data.jsonl \
  --detector-config-path output/detector_config.json \
  --score-cache-path output/Qwen3-4B-math/head_scores.pt \
  --all-experiments-path output/Qwen3-4B-math/all_experiments.jsonl \
  --top-k-values 1,2,4,8,16 \
  --window-sizes 1,3,5,7 \
  --aggregation weighted \
  --batch-size 1 \
  --max-seq-len 8192 \
  --dtype bfloat16 \
  --overwrite
```

The command produces:

- `head_scores.pt`, containing the calibrated score for each layer-head pair;
- `all_experiments.jsonl`, containing results for each evaluated
  `(top_k, window_size)` pair; and
- `detector_config.json`, containing the selected heads, aggregation rule,
  window size, and calibration statistics.

By default, non-leakage samples contribute a coefficient-of-variation
penalty. Pass `--disable-nonleak-penalty` to use cosine similarity for all
samples. Use `--from-cache` to reuse a compatible head-score cache and avoid
recomputing the first calibration stage.

## Threshold Calibration and Detector Evaluation

After selecting the heads and window size, calibrate the detection threshold
on a held-out validation split. The threshold is selected from an evenly
spaced grid, subject to the minimum span-recall constraint.

```bash
rigor-evaluate-detector \
  --model Qwen/Qwen3-4B \
  --model-key Qwen3-4B-math \
  --test-path examples/data/math/val_detector.jsonl \
  --detector-config-path output/detector_config.json \
  --threshold-tuning \
  --threshold-steps 200 \
  --min-span-recall 0.9 \
  --batch-size 1 \
  --max-seq-len 8192 \
  --dtype bfloat16
```

The selected threshold is written to the detector configuration. Threshold
selection must be performed on validation data rather than the final test
split.

To evaluate a saved threshold, omit `--threshold-tuning` and provide the final
test data:

```bash
rigor-evaluate-detector \
  --model Qwen/Qwen3-4B \
  --model-key Qwen3-4B-math \
  --test-path examples/data/math/test_detector.jsonl \
  --detector-config-path output/detector_config.json \
  --plot-dir output/Qwen3-4B-math/plots \
  --plot-lowest-n 7 \
  --batch-size 1 \
  --max-seq-len 8192 \
  --dtype bfloat16
```

Both modes report span precision, span recall, full recall, token precision,
F1, and the average number of predicted spans. Aggregate results and the
selected threshold are stored in the corresponding model entry of the
detector configuration.

To evaluate the included precalibrated detector without modifying the checked-in
configuration, first copy it to the output directory:

```bash
mkdir -p output
cp examples/detector_config.json output/detector_config.json
```

## Leakage-Safe Generation

RIGOR performs token-level leakage detection while the model is generating.
The privileged branch proposes candidate tokens. When the detector score
exceeds the calibrated threshold, RIGOR rolls back the local unsafe region and
samples a replacement token from the clean branch.

The following command runs leakage-safe generation with the included
Qwen3-4B mathematics detector:

```bash
rigor-generate \
  --input examples/data/math/test_detector.jsonl \
  --output output/generated.jsonl \
  --model Qwen/Qwen3-4B \
  --model-key Qwen3-4B-math \
  --detector-config examples/detector_config.json \
  --max-new-tokens 2048 \
  --temperature 0.7 \
  --top-p 1.0 \
  --do-sample \
  --enable-thinking \
  --limit 1 \
  --overwrite
```

### Generation Input Formats

The generator accepts any of the following JSONL structures.

Explicit clean and privileged prompts:

```json
{
  "prompt_wo_answer": "Question without privileged context",
  "privileged_context": "Privileged context",
  "prompt_w_answer": "Question with privileged context"
}
```

A privileged prompt and its exact context substring:

```json
{
  "prompt": "Question with privileged context",
  "privileged_context": "Privileged context"
}
```

A question and ground-truth answer:

```json
{
  "question": "Question",
  "ground_truth": "Answer"
}
```

For the third format, alternative field names can be specified with
`--question-field` and `--answer-field`.

Each output row preserves all input fields and adds:

```text
sample_index, prompt_wo_answer, prompt_w_answer, privileged_context,
text, token_ids, fixed, unfixed, repair_events
```

`repair_events` records each rollback and clean-branch replacement. `unfixed`
is `null` unless `--include-unfixed` is enabled.

## Reasoning-Trace Quality Evaluation

RIGOR includes three metrics for evaluating generated reasoning traces:

- **AB-PPL**: answer-blind rationale perplexity;
- **PCS**: privileged-context sensitivity; and
- **LF-GALG**: leakage-filtered ground-truth answer likelihood gain.

The evaluator requires an OpenAI-compatible completion endpoint, such as a
vLLM server, with token log-probabilities and `echo=True` support.

```bash
rigor-evaluate-traces \
  --input path/to/trace_data.jsonl \
  --output output/trace_quality.json \
  --model Qwen3-4B \
  --tokenizer Qwen/Qwen3-4B \
  --base-url http://localhost:8000/v1 \
  --api-key dummy \
  --num-proc 8 \
  --max-samples 1000 \
  --overwrite
```

Each trace-evaluation row must contain a `messages` array whose final element
is the assistant response:

```json
{
  "messages": [
    {"role": "user", "content": "Question"},
    {
      "role": "assistant",
      "reasoning_content": "Reasoning trace",
      "content": "Final answer: \\boxed{42}."
    }
  ],
  "ground_truth": "42"
}
```

Reasoning is read from `reasoning_content` when available. Otherwise, it is
extracted from the portion of `content` before `</think>`. If `ground_truth`
is absent, the evaluator attempts to parse it from `\boxed{...}` in the final
assistant response.

The output JSON contains aggregate summaries for all three metrics,
per-example scores, and the rendered prompts used for likelihood evaluation.

## Reproducibility

- Use the same model revision, tokenizer, and chat template across
  calibration, evaluation, and generation.
- The `--model-key` value must match a key in the detector configuration.
- Use separate data for head calibration, top-k/window calibration, threshold
  calibration, and final evaluation.
- Do not tune the detection threshold on the final test split.
- Use eager attention when an optimized attention backend does not return
  attention tensors.
- Set explicit random seeds and record all decoding parameters when reporting
  generation results.
