"""Per-object_subset logging of the per-sample loss terms."""

import sys
from pathlib import Path
from unittest.mock import patch

import torch

ANYTOP_ROOT = Path(__file__).resolve().parents[1]
if str(ANYTOP_ROOT) not in sys.path:
    sys.path.insert(0, str(ANYTOP_ROOT))

import train.training_loop as training_loop  # noqa: E402
from train.training_loop import TrainLoop  # noqa: E402


class _Tags:
    SUBSETS = {"Dog": "quadruped", "Cat": "quadruped", "Ghost": "drifting"}

    def object_subset_for(self, object_type):
        return self.SUBSETS.get(object_type)


def _stub():
    stub = TrainLoop.__new__(TrainLoop)
    stub._interval_loss_sums = {}
    stub._interval_loss_counts = {}
    stub._subset_names = []
    stub._subset_index_of_species = {}
    stub._subset_loss_sums = {}
    stub._subset_sample_counts = {}
    return stub


def test_subset_metrics_average_each_term_over_its_own_samples():
    stub = _stub()
    with patch.object(training_loop, "dataset_tags", return_value=_Tags()):
        stub._accumulate_subset_losses(
            {"dpos_loss": torch.tensor([1.0, 3.0, 10.0]), "l_simple": torch.tensor([5.0, 5.0, 5.0])},
            ["Dog", "Cat", "Ghost"],
        )
        stub._accumulate_subset_losses(
            {"dpos_loss": torch.tensor([20.0, 2.0])},
            ["Ghost", "Bird"],
        )
    metrics = stub._flush_interval_loss_metrics()

    assert metrics["subset/quadruped/dpos_loss"] == 2.0
    assert metrics["subset/drifting/dpos_loss"] == 15.0
    assert metrics["subset/untagged/dpos_loss"] == 2.0
    # Only the listed terms are split by subset.
    assert not any("l_simple" in key for key in metrics)
    # A flush starts the next interval empty.
    assert stub._flush_interval_loss_metrics() == {}
