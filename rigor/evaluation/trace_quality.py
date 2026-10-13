"""Evaluate reasoning traces with AB-PPL, PCS, and LF-GALG."""

from __future__ import annotations

import argparse
import json
import os
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from numpy import exp, mean, median, percentile

PCS_TOP_RATIO = 0.05
PRIVILEGED_CONTEXT_TEMPLATE = (
    "\n\nGiven the ground truth answer is $\\boxed{{{answer}}}$."
)
CORRECTNESS_SYSTEM_PROMPT = (
    "You are a precise problem solver. Solve step by step, then give the "
    "final answer on its own line."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Model name served by vLLM.")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path or ID; defaults to --model.",
    )
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--pcs-top-ratio", type=float, default=PCS_TOP_RATIO)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_boxed_answer(text):
    marker = r"\boxed{"
    start = text.find(marker)
    if start == -1:
        return None
    start += len(marker)
    if start < len(text) - 1 and text[start] == "}" and text[start + 1] == "}":
        return "}"
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index]
    return None


def replace_boxed(text):
    def replacer(match):
        answer = parse_boxed_answer(match.group(0))
        return match.group(0) if not answer else str(answer)

    return re.sub(r"\\boxed\{([^{}]*)\}", replacer, text)


def get_ground_truth(example):
    answer = example.get(
        "ground_truth",
        parse_boxed_answer(example["messages"][-1]["content"]),
    )
    if not answer:
        return None
    answer = str(answer)
    if "another answer" in answer.lower():
        return None
    return " " + (parse_boxed_answer(answer) or answer).strip()


def get_reasoning_content(example):
    assistant = example["messages"][-1]
    if assistant.get("reasoning_content"):
        return assistant["reasoning_content"]

    reasoning = assistant["content"].split("</think>")[0]
    if reasoning.startswith("<think>/n"):
        reasoning = reasoning.split("<think>/n")[-1]
    elif reasoning.startswith("<think>"):
        reasoning = reasoning.split("<think>")[-1]
    return reasoning


def get_question_messages(example, correctness_system=False):
    messages = deepcopy(example["messages"][:-1])
    if correctness_system:
        system = {"role": "system", "content": CORRECTNESS_SYSTEM_PROMPT}
        if messages[0]["role"] != "system":
            messages = [system, *messages]
        else:
            messages[0] = system
    return messages


def add_privileged_context(messages, answer):
    messages = deepcopy(messages)
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "user":
            messages[index]["content"] += PRIVILEGED_CONTEXT_TEMPLATE.format(
                answer=answer
            )
            return messages
    raise ValueError("No user message found for privileged-context insertion.")


@dataclass
class TraceQualityEvaluator:
    model: str
    base_url: str
    api_key: str
    tokenizer: object
    pcs_top_ratio: float = PCS_TOP_RATIO

    def client(self):
        from openai import OpenAI

        return OpenAI(base_url=self.base_url, api_key=self.api_key)

    def reasoning_prompt(self, messages):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if "<think>" not in prompt:
            prompt += "<think>\n"
        return prompt

    def answer_prompt(self, messages, reasoning_content=None):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if reasoning_content is None:
            prompt += "</think>\nFinal answer:" if "<think>" in prompt else "Final answer:"
        elif "<think>" in prompt:
            prompt += reasoning_content + "\n</think>\nFinal answer:"
        else:
            prompt += "<think>\n" + reasoning_content + "\n</think>\nFinal answer:"
        return replace_boxed(prompt)

    def completion_logps(
        self,
        prompt,
        completion,
        return_tokens=False,
        client=None,
    ):
        prompt_tokens = len(self.tokenizer.encode(prompt))
        try:
            client = client or self.client()
            response = client.completions.create(
                model=self.model,
                prompt=prompt + completion,
                max_tokens=1,
                logprobs=1,
                echo=True,
            )
            logprobs = response.choices[0].logprobs
            values = logprobs.token_logprobs[prompt_tokens:]
            tokens = logprobs.tokens[prompt_tokens:]
            if values:
                values = values[:-1]
                tokens = tokens[:-1]
            pairs = [
                (token, value)
                for token, value in zip(tokens, values)
                if value is not None
            ]
            if return_tokens:
                return [value for _, value in pairs], [token for token, _ in pairs]
            return [value for _, value in pairs]
        except Exception as error:
            print("Error:", error)
            return ([], []) if return_tokens else []

    def fluency(self, example):
        try:
            messages = get_question_messages(example)
        except Exception:
            return {"rendered_chat_fluency": "", "ab_ppl": None}
        reasoning = get_reasoning_content(example)
        prompt = self.reasoning_prompt(messages)
        logps = self.completion_logps(prompt, reasoning, client=self.client())
        if not logps:
            return {"rendered_chat_fluency": "", "ab_ppl": None}
        return {
            "rendered_chat_fluency": prompt + reasoning,
            "ab_ppl": float(exp(-mean(logps))),
        }

    def pcs(self, example):
        empty = {
            "rendered_chat_pcs_blind": "",
            "rendered_chat_pcs_privileged": "",
            "pcs": None,
            "pcs_filtered_reasoning": None,
            "pcs_removed_fraction": None,
        }
        answer = get_ground_truth(example)
        if answer is None:
            return empty
        reasoning = get_reasoning_content(example)
        try:
            blind_messages = get_question_messages(example)
        except Exception:
            return empty

        privileged_messages = add_privileged_context(blind_messages, answer)
        blind_prompt = self.reasoning_prompt(blind_messages)
        privileged_prompt = self.reasoning_prompt(privileged_messages)
        client = self.client()
        blind_logps, blind_tokens = self.completion_logps(
            blind_prompt,
            reasoning,
            return_tokens=True,
            client=client,
        )
        privileged_logps, _ = self.completion_logps(
            privileged_prompt,
            reasoning,
            return_tokens=True,
            client=client,
        )
        if (
            not blind_logps
            or not privileged_logps
            or len(blind_logps) != len(privileged_logps)
        ):
            return {
                **empty,
                "rendered_chat_pcs_blind": blind_prompt + reasoning,
                "rendered_chat_pcs_privileged": privileged_prompt + reasoning,
            }

        deltas = [
            privileged - blind
            for privileged, blind in zip(privileged_logps, blind_logps)
        ]
        positive_deltas = [max(0.0, value) for value in deltas]
        k = max(1, int(len(positive_deltas) * self.pcs_top_ratio + 0.999999))
        pcs = mean(sorted(positive_deltas, reverse=True)[:k])
        positive_indices = [index for index, value in enumerate(deltas) if value > 0]
        removed = set(
            sorted(positive_indices, key=lambda index: deltas[index], reverse=True)[:k]
        )
        filtered_reasoning = "".join(
            token for index, token in enumerate(blind_tokens) if index not in removed
        )
        return {
            "rendered_chat_pcs_blind": blind_prompt + reasoning,
            "rendered_chat_pcs_privileged": privileged_prompt + reasoning,
            "pcs": float(pcs),
            "pcs_filtered_reasoning": filtered_reasoning,
            "pcs_removed_fraction": float(len(removed) / len(deltas)),
        }

    def leakage_filtered_gain(self, example):
        empty = {
            "rendered_chat_lf_galg_base": "",
            "rendered_chat_lf_galg_reason": "",
            "lf_galg": None,
        }
        answer = get_ground_truth(example)
        filtered_reasoning = example.get("pcs_filtered_reasoning")
        if answer is None or filtered_reasoning is None:
            return empty
        try:
            messages = get_question_messages(example, correctness_system=True)
        except Exception:
            return empty

        base_prompt = self.answer_prompt(messages)
        filtered_prompt = self.answer_prompt(messages, filtered_reasoning)
        client = self.client()
        base_logps = self.completion_logps(base_prompt, answer, client=client)
        filtered_logps = self.completion_logps(
            filtered_prompt,
            answer,
            client=client,
        )
        lf_galg = None
        if base_logps and len(base_logps) == len(filtered_logps):
            lf_galg = mean(filtered_logps) - mean(base_logps)

        return {
            "rendered_chat_lf_galg_base": base_prompt + answer,
            "rendered_chat_lf_galg_reason": filtered_prompt + answer,
            "lf_galg": None if lf_galg is None else float(lf_galg),
        }


def summarize_robust(values):
    values = [value for value in values if value is not None]
    q25 = percentile(values, 25)
    q50 = median(values)
    q75 = percentile(values, 75)
    return {
        "mean": float(mean(values)),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "n": len(values),
    }


def build_results(dataset):
    return {
        "ab_ppl": summarize_robust(dataset["ab_ppl"]),
        "pcs": summarize_robust(dataset["pcs"]),
        "lf_galg": summarize_robust(dataset["lf_galg"]),
        "per_example": [
            {"ab_ppl": ab_ppl, "pcs": pcs, "lf_galg": lf_galg}
            for ab_ppl, pcs, lf_galg in zip(
                dataset["ab_ppl"],
                dataset["pcs"],
                dataset["lf_galg"],
            )
        ],
        "rendered_chat_fluency": [
            {"text": text, "ab_ppl": ab_ppl}
            for text, ab_ppl in zip(
                dataset["rendered_chat_fluency"],
                dataset["ab_ppl"],
            )
        ],
        "rendered_chat_pcs": [
            {
                "blind": blind,
                "privileged": privileged,
                "pcs": pcs,
                "filtered_reasoning": filtered,
                "removed_fraction": fraction,
            }
            for blind, privileged, pcs, filtered, fraction in zip(
                dataset["rendered_chat_pcs_blind"],
                dataset["rendered_chat_pcs_privileged"],
                dataset["pcs"],
                dataset["pcs_filtered_reasoning"],
                dataset["pcs_removed_fraction"],
            )
        ],
        "rendered_chat_lf_galg": [
            {
                "base": base,
                "filtered_reason": filtered,
                "lf_galg": lf_galg,
            }
            for base, filtered, lf_galg in zip(
                dataset["rendered_chat_lf_galg_base"],
                dataset["rendered_chat_lf_galg_reason"],
                dataset["lf_galg"],
            )
        ],
    }


def main():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists. Pass --overwrite.")
    if args.num_proc < 1 or args.max_samples < 1:
        raise ValueError("--num-proc and --max-samples must be positive.")
    if not 0 < args.pcs_top_ratio <= 1:
        raise ValueError("--pcs-top-ratio must be in (0, 1].")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or "dummy"
    evaluator = TraceQualityEvaluator(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer or args.model),
        pcs_top_ratio=args.pcs_top_ratio,
    )
    dataset = load_dataset(
        "json",
        data_files=str(args.input),
        split=f"train[:{args.max_samples}]",
    )
    dataset = dataset.map(evaluator.fluency, num_proc=args.num_proc)
    dataset = dataset.map(evaluator.pcs, num_proc=args.num_proc)
    dataset = dataset.map(evaluator.leakage_filtered_gain, num_proc=args.num_proc)
    results = build_results(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results), encoding="utf-8")
    for metric in ("ab_ppl", "pcs", "lf_galg"):
        summary = results[metric]
        print(
            f"{metric}: mean={summary['mean']:.6g} "
            f"median={summary['median']:.6g} n={summary['n']}"
        )
    print(f"Outputs saved at {args.output}")


if __name__ == "__main__":
    main()
