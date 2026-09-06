# AnyTop: Character Animation Diffusion with Any Topology

The official PyTorch implementation of the paper [**"AnyTop: Character Animation Diffusion with Any Topology"**]().

Please visit our [**webpage**](https://anytop2025.github.io/Anytop-page/) for more details.

![teaser](https://github.com/Anytop2025/Anytop-page/blob/main/static/videos/anytop_teaser/teaser.gif)

---

## Fork Improvements

This fork includes significant improvements over the original repository while preserving the core Gaussian Diffusion architecture:

### Data Quality & Preprocessing
- **Semantic Joint Name Embeddings** — T5-encoded joint names with capitalization, left/right word-order, and compound-word normalization. Cached in `cond.npy`.
- **Canonical Joint Naming** — Normalized joint names, inferred symmetry pairs, end-effectors, and contact joints from semantic name tokens.
- **Loop Detection** — Automatic detection of looping motion clips.
- **Preprocessing + Validation Workflow** — Unified CLI (`preprocess_and_validate.py`) chains preprocessing with immediate validation.

### Training Enhancements
- **EMA Model Averaging** — `--use_ema` for improved generalization.
- **StepLR Scheduler** — Configurable learning rate decay (`--lr_scheduler_step_size`, `--lr_scheduler_gamma`).
- **Balanced Sampler** — `--balanced` for fair sampling across topologies.

### Evaluation
- **Distribution-Based Motion Quality Scorer** — Low-shot weighted-reference evaluation without autoencoders or discriminators. Scores macro distribution fidelity and local joint naturalness.
- **Action Group Split** — Train one model per action group: `--action_group locomotion|stationary|transition` is required at training and takes exactly one of the three (no `all`, no list). The group is recorded in the checkpoint's `args.json`, and generation has no `--action_group` flag at all — it reads the group from there, so a checkpoint can only ever be sampled as the group it was trained on.
- **Text-to-Motion Conditioning** — `--action_label_cond` conditions on the frozen-T5 embedding of the clip's `action_label`, which is controlled keywords in canonical order (`"run, forward, left, fast"`); `--action_label "run, forward"` at generation time, with `--action_label_cfg_scale` to amplify it.
- **Semantic Joint Groups** — Automatic root/axial/limbs grouping from skeleton metadata for per-group evaluation.

### Data Loading
- **Motion LRU Cache** — In-memory cache for raw motion clips (`--motion_cache_size`).
- **Background Prefetch Loader** — Overlaps I/O with GPU compute (`--main_process_prefetch_batches`).

### Performance & Robustness
- **Selective BF16 Training** — BF16 autocast on Linear/Attention/Conv modules only; softmax and dropout remain in FP32. Use `--amp_dtype bf16` to enable.
- **Geodesic Loss Stability** — `atan2`-based rotation distance replaces `acos`, eliminating gradient blow-up near identity rotations.
- **Safe 6D Rotation** — Fallback mechanism for near-collinear 6D rotation vectors.
- **Deterministic Resume** — Full RNG state (torch/cuda/python/numpy) saved and restored for reproducible data shuffling.
- **Gradient Norm Hard-Fail** — Automatic abort if `grad_norm > 1e12` prevents silent training corruption.
- **Optimizer State Sanitization** — Non-finite optimizer slots detected and cleaned after resume.

---

## Update Notice

📢 September 25, 2025 – Important bug fix related to dataset preprocessing and handling unseen motions. If you are working with either, please pull the latest commits and rerun the preprocessing procedure.   
📢 June 2, 2025 – Blender visualization script released.   
📢 May 31, 2025 – Evaluation code uploaded.  
📢 April 27, 2025 – New models uploaded (minor bug fix) — Update your model paths.  
📢 April 27, 2025 – New cond.npy uploaded — Override your local file if you have already created the dataset.
  * To handle both updates above, simply remove the current cond.npy file from your dataset directory and re-run "Download Pretrained Models and Dataset Dependencies."
    
## Release Timeline

✅ April 6, 2025 – Training & inference code & preprocessing code  
✅ April 12, 2025 – Pretrained models  
✅ April 27, 2025 – DIFT feature correspondence code  
✅ May 31, 2025 – Evaluation code  
✅ June 2, 2025 – Rendering code  
📌 *(Processed dataset temporarily withheld due to licensing clarification)*  

## Getting started

This code was tested on `Ubuntu 18.04.5 LTS` and requires:

* Python 3.8
* conda3 or miniconda3
* CUDA capable GPU (one is enough)

### 1. Setup environment
Setup conda env:
```shell
conda env create -f environment.yaml
conda activate anytop
```

### 2. Download and Preprocess Truebones dataset
Due to ongoing licensing clarification, we are currently not planning to publish the processed dataset. 
However, we provide everything you need to process it yourself using our preprocessing script:
(a) Download the full dataset from the [official Truebones website](https://truebones.gumroad.com/l/skZMC) 
(b) Place the Truebone_Z-OO directory inside our repository under ./datasets/truebones/zoo/
(c) Run the following command to begin preprocessing (it preprocesses every object found in the raw data directory, then validates the result):
```shell
python preprocess_and_validate.py
```
Use `--filter` to (re)process only a subset of objects incrementally, e.g. `python preprocess_and_validate.py --filter "Horse,Raptor*"`.

Note: The preprocessing may take several hours to complete, primarily due to inverse kinematics calculations.

### 3. Download Pretraind Models and dataset dependencies
Download pretrained models to ./save dorectory by running the following command:
```shell
python -m utils.download_dependencies
```

## Preprocessing new skeleton 
In addition to providing the preprocessing code for the full Truebones dataset, we also guide you through applying our pipeline to in-the-wild skeletons from any source-not just Truebones. This is useful for adapting new skeletons to our system, whether for inference on unseen characters or for training with alternative datasets.
While the preprocessing code was designed to be as generic and adaptable as possible, some skeleton-specific adjustments may still be required, as the pipeline was originally tailored to Truebones. For instance, it uses indicative joint name substrings for foot classification and predefined velocity/height thresholds for foot contact detection-heuristics that have worked well in our experiments with Truebones.
That said, we've tested the pipeline on BVH files from Mixamo and other sources to ensure its generalizability across different skeleton formats.

The script (`tools/process_new_skeleton.py`) accepts the following input arguments:
*tpos_path* - An FBX/GLB/GLTF file whose bind/rest pose defines the NPY encoding base (required).
*save_dir* - Output directory (required).
*object_type* - Species/type name (e.g. "Dragon"). Inferred from the tpos-path filename when omitted.
*species_tags* - Comma-separated species tags (motion descriptor) for the object type, e.g. 'Quadruped,Large,Lumbering'. REQUIRED: it defines the descriptor baked into cond.npy. There is no fallback to the default dataset's species_tags.jsonl.
*crop_enabled* - Enable skeleton cropping to MAX_JOINTS=100. Off by default (inference has no joint cap); enable for training-compatible preprocessing.
*reference_cond_path* - cond.npy to inherit the per-object_subset standardization statistics from. REQUIRED: the stats belong to a trained checkpoint, so pass the checkpoint's own cond.npy snapshot (there is no fallback to the processed dataset directory).
*skip_t5_embeddings* - Skip T5 embedding computation (the caller injects them via attach_t5_embeddings_to_cond).
*yes* - Skip all interactive confirmation prompts (headless / automated calls).

Finally, you can run the command: 

```shell
python tools/process_new_skeleton.py --tpos-path assets/Chicken_Tpose.glb --save-dir outputs/new_skeleton/Chicken --object-type Chicken --species-tags "Quadruped,Heavy,Lumbering" --reference-cond-path save/<checkpoint>/cond.npy
```

The generated cond.npy is designed for **inference** -- pass it via `--cond-path` to `generate.py`. It is not suitable for training unless `--crop-enabled` is set.

The code will create the following under save_dir:
save_dir/
        |_motions
        |_bvhs
        cond.npy
        species_tags.jsonl
1. cond.npy contains the skeleton representation (joint name embeddings, graph conditions, canonical feature-space metadata), which is the required input for motion synthesis via `--cond-path`.
2. species_tags.jsonl is the sidecar carrying the object type's motion descriptor; cond.npy bakes its `species_tags` field from it.
3. motions/ and bvhs/ are created but hold no clips for a rest-pose-only build (there is no source motion to process).
       
       
## Motion Synthesis

```bat
generate.bat --object_type <skeleton_name>
```

The full argument set of the current run lives in `generate.bat`; every flag is
a plain CLI option of `python sample/generate.py --help`. It resolves the latest
checkpoint under `save/merged_locomotion_v3/` (the `RUN_NAME` set in the script)
automatically and passes the target skeleton plus any extra flags through to
`sample/generate.py` (e.g. `generate.bat --object_type
Horse --loop --action_label "run, forward"`).

## Train AnyTop 

```bat
train.bat
```

The full argument set of the current run lives in `train.bat`; every flag is a
plain CLI option of `python train/train_anytop.py --help`.

## Acknowledgments
We want to thank the following contributors that our code is based on:
[mdm](https://github.com/GuyTevet/motion-diffusion-model), [GRPE](https://github.com/lenscloth/GRPE/tree/master), [audiocraft](https://github.com/facebookresearch/audiocraft)

## License
This code is distributed under an [MIT LICENSE](LICENSE).
Note that our code depends on other libraries that have their own respective licenses that must also be followed.
