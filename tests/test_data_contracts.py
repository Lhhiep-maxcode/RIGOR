from argparse import Namespace

from rigor.evaluation.trace_quality import parse_boxed_answer
from rigor.generation.generate import prepare_prompt


def test_generation_question_answer_contract():
    args = Namespace(question_field="question", answer_field="ground_truth")
    prompt = prepare_prompt(
        {"question": "What is 1 + 1?", "ground_truth": "2"},
        args,
    )
    assert prompt == {
        "clean_prompt": "What is 1 + 1?\n\nExplain your solution step by step.",
        "privileged_context": " Given the ground truth answer is 2",
        "privileged_prompt": (
            "What is 1 + 1?\n\nExplain your solution step by step. "
            "Given the ground truth answer is 2"
        ),
    }


def test_nested_boxed_answer():
    assert parse_boxed_answer(r"Result: \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
