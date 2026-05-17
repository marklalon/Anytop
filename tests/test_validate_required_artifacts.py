import os
import sys
from pathlib import Path

import numpy as np
import pytest


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ANYTOP_ROOT = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_ANYTOP_ROOT)

for _path in [_REPO_ROOT, _ANYTOP_ROOT, os.path.join(_ANYTOP_ROOT, 'utils')]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


from validate_anytop_dataset import (  # noqa: E402
    BVHS_DIR,
    MOTION_DIR,
    STRUCTURAL_NORM_PRIORS_FILE,
    ValidationError,
    _read_required_artifacts,
)


def _build_min_dataset(dataset_dir: Path) -> None:
    (dataset_dir / MOTION_DIR).mkdir(parents=True)
    (dataset_dir / BVHS_DIR).mkdir(parents=True)
    np.save(dataset_dir / 'cond.npy', {})
    (dataset_dir / 'metadata.txt').write_text('', encoding='utf-8')
    (dataset_dir / 'positions_error_rate.txt').write_text(
        'Position squared error per source clip:\n',
        encoding='utf-8',
    )


def test_read_required_artifacts_requires_structural_prior_bank(tmp_path):
    _build_min_dataset(tmp_path)

    with pytest.raises(ValidationError, match=STRUCTURAL_NORM_PRIORS_FILE):
        _read_required_artifacts(tmp_path, silent=True)


def test_read_required_artifacts_accepts_structural_prior_bank(tmp_path):
    _build_min_dataset(tmp_path)
    prior_bank_path = tmp_path / STRUCTURAL_NORM_PRIORS_FILE
    np.save(prior_bank_path, {'schema_version': 3})

    artifacts = _read_required_artifacts(tmp_path, silent=True)

    assert artifacts[-1] == prior_bank_path