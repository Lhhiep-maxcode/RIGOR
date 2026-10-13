import torch

from rigor.detector.calibrate import (
    coefficient_variation,
    cosine,
    rolling_max,
    token_spans,
)
from rigor.detector.evaluate import metric_row, pooled_threshold_metrics


def test_detector_math_and_span_metrics():
    real = torch.tensor([0.1, 0.8, 0.7, 0.1])
    ideal = torch.tensor([0.0, 1.0, 1.0, 0.0])

    assert torch.isclose(cosine(real, ideal), torch.tensor(1.5) / (real.norm() * ideal.norm()))
    assert torch.isclose(
        coefficient_variation(torch.tensor([1.0, 1.0])),
        torch.tensor(0.0),
    )
    assert rolling_max(real, 3).shape == real.shape
    assert token_spans(ideal.bool()) == [{"start_token": 1, "end_token": 3}]

    stats, prediction = metric_row(real, ideal, threshold=0.5)
    assert prediction.tolist() == [False, True, True, False]
    assert stats["precision"] == 1.0
    assert stats["recall"] == 1.0

    pooled = pooled_threshold_metrics(
        [{"real": real, "ideal": ideal}],
        threshold=0.5,
    )
    assert pooled["f1"] == 1.0
    assert pooled["token_precision"] == 1.0
