# AnyTop Preprocessing + Validation Workflow

## Overview
A unified command-line interface that chains AnyTop dataset preprocessing directly with validation. This ensures both steps run in sequence and provides immediate feedback on data quality.

## Quick Start

### Python (All Platforms)
```bash
# Run full workflow (preprocess + corrupted references + validate)
python preprocess_and_validate.py

# Use more CPU across object types and BVH files
python preprocess_and_validate.py --objects-subset quadropeds --object-workers 8 --file-workers 8

# Regenerate stored corrupted references only
python tools/export_corrupted_truebones_samples.py --objects-subset quadropeds_clean

# Render a random QA subset of stored corrupted references
python tools/render_corrupted_truebones_previews.py --output-dir outputs/corrupted_previews --objects-subset quadropeds_clean --sample-limit 6 --random-seed 1234

# Validate only (skip preprocessing)
python preprocess_and_validate.py --validate-only

# Skip validation (faster, for CI/testing)
python preprocess_and_validate.py --skip-validate
```

## What It Does

### Step 1: Preprocessing
- Runs `utils/create_dataset.py`
- Creates motion `.npy` tensors from input BVH files
- Generates `cond.npy` (conditioning file with skeleton metadata)
- Outputs summary to `metadata.txt` and error rates to `positions_error_rate.txt`

### Step 2: Stored Corrupted References
- Runs `tools/export_corrupted_truebones_samples.py`
- Writes corrupted references next to `motions/` under `corrupted_references/`
- Saves only `.npy` + `.json` metadata files for speed
- Does not generate MP4 previews during export

### Step 3: Validation
- Runs `tests/check_anytop_dataset.py`
- Verifies required artifacts exist
- Validates all preprocessed motion/BVH files by default
- Checks data types, shapes, paired stems, and finite values
- Catches preprocessing issues before training

## Options

| Option | Effect |
|--------|--------|
| `--validate-only` | Skip preprocessing, run validation on existing dataset |
| `--skip-validate` | Skip validation (useful for quick checks or CI) |
| `--skip-corrupted-export` | Skip generating stored corrupted references after preprocessing |
| `--objects-subset` | Expected object type subset (`all`, `hound`, `chicken`, etc.) |
| `--object-workers` | Concurrent characters to preprocess |
| `--file-workers` | Worker threads per character for BVH file processing |
| `--sample-count` | Limit file validation to first `N` motions/BVHs; `0` means validate all files |
| `--corrupted-seed` | Random seed used for stored corrupted references |
| `--corrupted-sample-limit` | Limit number of motions that get corrupted references |

## Exit Codes
- `0` - Success (both preprocessing and validation passed)
- `1` - Failure (preprocessing or validation failed)

## Output Location
By default, preprocessed data is saved to the directory specified by:
- `--dataset-dir` command-line argument, or
- `data/` (if environment variable not set)

Stored corrupted references are written beside clean motions under:
- `corrupted_references/`

## Troubleshooting

### `ModuleNotFoundError` during validation
The validation step requires the same Python environment used for preprocessing. Make sure to:
```bash
# Activate the project's virtual environment
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Then run the workflow from Anytop directory
cd Anytop
python preprocess_and_validate.py
```

Validation now checks only the preprocessed files and does not import the dataloader.

### Preprocessing succeeds but validation fails
Common causes:
1. **Shape mismatches** - Motion tensors don't match skeleton definitions
2. **NaN/Inf values** - Invalid floating-point data in tensors
3. **File counts** - Unequal number of `.npy` and `.bvh` files

Run validation directly for detailed file-by-file diagnostics:
```bash
python preprocess_and_validate.py --validate-only
```

## Files Involved

### New Files
- `preprocess_and_validate.py` - Main orchestration script (Python)
- `tests/check_anytop.py` - Environment compatibility check
- `tests/check_anytop_dataset.py` - Dataset validation script
- `tools/export_corrupted_truebones_samples.py` - Stored corrupted-reference exporter
- `tools/render_corrupted_truebones_previews.py` - Optional MP4 preview renderer

### Existing Referenced Files
- `utils/create_dataset.py` - Dataset creation entry point
- `data_loaders/truebones/truebones_utils/motion_process.py` - Actual preprocessing logic
- `data_loaders/get_data.py` - Dataloader instantiation

## Advanced Usage

### Integration with CI/CD
For automated pipelines, file validation is safe by default. Use `--skip-validate` only if you want to skip the validation step entirely:
```bash
python preprocess_and_validate.py --skip-validate
```

### Validate Specific Subset
If you only preprocessed a subset of animals:
```bash
python preprocess_and_validate.py --objects-subset chicken
```

### Selective Validation
Validate without preprocessing:
```bash
python preprocess_and_validate.py --validate-only
```
This is useful after manual preprocessing or when revalidating previous runs.
