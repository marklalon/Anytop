"""GraphMotionDecoder: per-batch relation tables and the shared loop phase table."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import model.motion_transformer as motion_transformer_module  # noqa: E402
from model.motion_transformer import (  # noqa: E402
    GraphMotionDecoder,
    GraphMotionDecoderLayer,
)


D, H = 16, 4
T, J = 5, 6


def test_decoder_expands_graph_relations_once_per_batch_not_per_frame():
    layer = GraphMotionDecoderLayer(D, H, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(layer, num_layers=1)
    captured = {}

    def stub(output, *a, **kw):
        captured["topology_shape"] = a[1].shape
        captured["edge_shape"] = a[2].shape
        return output

    dec.layers[0].forward = stub
    y = {
        "graph_dist": torch.zeros(2, 3, 3, dtype=torch.int64),
        "joints_relations": torch.zeros(2, 3, 3, dtype=torch.int64),
    }

    dec.forward(
        tgt=torch.zeros(5, 2, 3, D),
        timesteps_embs=torch.zeros(2, D),
        memory=None,
        y=y,
    )

    assert captured["topology_shape"] == (2, H, 3, 3)
    assert captured["edge_shape"] == (2, H, 3, 3)


def test_decoder_reuses_precomputed_loop_phase_embeddings(monkeypatch):
    calls: list[tuple[int, int]] = []
    original = motion_transformer_module.circular_phase_embedding

    def wrapped(length, dim, device, dtype):
        calls.append((length, dim))
        return original(length, dim, device, dtype)

    monkeypatch.setattr(motion_transformer_module, "circular_phase_embedding", wrapped)

    layer = GraphMotionDecoderLayer(D, H, dim_feedforward=32, dropout=0.0)
    dec = GraphMotionDecoder(layer, num_layers=3)
    seen = []

    def stub(output, *a, **kw):
        seen.append(kw["loop_phase_embedding"])
        return output

    for decoder_layer in dec.layers:
        decoder_layer.forward = stub

    y = {
        "graph_dist": torch.zeros(2, J, J, dtype=torch.int64),
        "joints_relations": torch.zeros(2, J, J, dtype=torch.int64),
    }
    dec.forward(
        tgt=torch.zeros(T, 2, J, D),
        timesteps_embs=torch.zeros(2, D),
        memory=None,
        y=y,
        loop_phase_mask=torch.tensor([True, False]),
    )

    # One circular table at d_model, shared by every layer.
    assert calls == [(T, D)]
    assert len({id(phase) for phase in seen}) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
