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
- **Text-to-Motion Conditioning** — `--action_label_cond` conditions on the clip's `action_label` (frozen T5 embedding + a group-masked multi-hot over the controlled action vocabulary); `--action_label "run"` at generation time.
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

The script accepts the following input Arguments:
*object_name* - A character's indicative name (e.g., "Dog"). If the skeleton you wish to process is outside of Truebones dataset, 
please make sure the object name you choose does not already exist in Truebones charachters names. 
All of Truebones characters names can be found in Truebones_skeletons.txt file. 
*bvh_dir* - Directory containing BVH files of the skeleton. More files improve statistical accuracy for motion denormalization.
*save_dir* - Output directory.
*tpos_bvh* - A BVH file of the character's natural rest pose for meaningful rotation learning. 
If missing, the code will search for a good candidate pose from the provided BVH files. 

Finally, you can run the command: 

```shell
python -m utils.process_new_skeleton --object_name Chicken --bvh_dir assets/Truebones_Chicken --save_dir dataset/truebones/zoo/Chicken --tpos_bvh assets/Truebones_Chicken/Chicken_TPOSE.bvh
```

The code will create the following under save_dir:
save_dir/
        |_motions
        |_animations
        |_bvhs
        cond.npy
1. In motions directory, you will find npy files, which are the processed motion features of each bvh file. 
This is useful in case you would like to use this data for training.  Note that motions longer than 240 frames wil be splited into separate npy files (this statement holds for the following outputs as well). 
2. In animation directory, you will find mp4 files corresponding to each of the processed bvhs.  
This is a good sanity check that everything worked as expected.
3. In bvhs dir you can find the processed bvhs (with the new orientation, scale etc.)
4. cond.npy contains the skeletons representation, including joints names ambeddings and graph conditions, which is a required input for motion synthesis. 
       
       
## Motion Synthesis

### Generate motion for skeleton from Truebones dataset
We categorize Truebones skeletons by body plan into the following `--object_subsets`: `quadruped`, `biped`, `multiped`, `serpentine`, `aquatic`, and `winged`.
In addition to a unified model trained on the entire dataset, we also trained specialized models per body plan.

The skeleton-to-subset mapping is derived from the first column (body plan) of `dataset/truebones/zoo/truebones_processed/species_tags.jsonl`, which is the single source of truth for both the per-species motion descriptor and the `--object_subsets` groupings.

If you'd like to synthesize motion using our pre-trained models, ensure that all model checkpoint files are located in the ./save directory (this should already be the case if you've completed Step 3: Download Pretrained Models).

For example, to generate motion using models trained on winged objects, you can synthesize motion for one or more skeletons from the `winged` subset using the following command:

```shell
python -m sample.generate  --model_path save/flying_model_dataset_truebones_bs_16_latentdim_128/model000229999.pt --object_type Parrot2 Bat --num_repetitions 3
```
As the code is fully generic, you can generate motion for unseen skeletons (that do not belong to the subset the model was trained on) using the exact same syntax. 
For example, you can explore synthesizing motions for the Ostrich skeleton using the `winged` subset model:
```shell
python -m sample.generate  --model_path save/flying_model_dataset_truebones_bs_16_latentdim_128/model000229999.pt --object_type Tyranno --num_repetitions 3
```

### Generate unseen skeleton outside of Truebones dataset
We support motion synthesis for skeletons outside of Truebones dataset, provided as bvh file/s. 
To do that, you must first run our pre-processing pipeline on the new skeleton to create cond.py file for the skeleton, as described in 
Preprocessing new skeleton section above. Once you've accomplish this part, you can synthesize motion of the new skeleton by running the command:

```shell
python -m sample.generate  --model_path <model_path> --object_type <skeleton_name> --num_repetitions 3 --cond_path <path_to_cond_npy_file>
```

**You may also define:**
* `--device` id.
* `--seed` to sample different seeds.
* `--num_frames` number of frames in the sampled motion (default 60).

**Running those will get you:**

* `<object_type>_rep_<rep_id>_#<sample_id>.npy` file with xyz positions of the generated animation
* `<object_type>_rep_<rep_id>_#<sample_id>.mp4` a stick figure animation for each generated motion
* `<object_type>_rep_<rep_id>_#<sample_id>.bvh` bvh file of the generated motion

Stick figure animation looks something like this:

![example]( assets/smaller_stick_fig.gif )

## Train AnyTop 

To reproduce the unified paper model, run:
```shell
python -m train.train_anytop --model_prefix all --objects_subset all --lambda_geo 1.0 --auto_resume --balanced
```

To reproduce the bipeds paper model, run:
```shell
python -m train.train_anytop --model_prefix biped --objects_subset biped --lambda_geo 1.0 --auto_resume --balanced
```

To reproduce the quadrupeds paper model, run:
```shell
python -m train.train_anytop --model_prefix quadruped --objects_subset quadruped --lambda_geo 1.0 --auto_resume --balanced
```
To reproduce the multiped paper model, run (snakes are now a separate `serpentine` subset rather than grouped in):
```shell
python -m train.train_anytop --model_prefix multiped --objects_subset multiped --lambda_geo 1.0 --auto_resume --balanced
```

To reproduce the flying animals paper model, run:
```shell
python -m train.train_anytop --model_prefix winged --objects_subset winged --lambda_geo 1.0 --auto_resume --balanced
```
* **General instructions** Checkout './utils/parser_utils.py' to view all configurable parameters and default settings. '--balanced' flag is used to activate the balancing sampler, ensuring fair sampling of all skeletons. Use '--auto_resume' if you want the script to continue from the latest checkpoint in save_dir. Without it, training starts fresh and overwrites existing checkpoints in save_dir. 
* `--action_group` is **required** and picks the one group to train on. Each clip belongs to exactly one of `locomotion` (sustained displacement), `stationary` (in-place / interactive) or `transition` (pose changes), and each group trains its own model — so the flag takes a single group, never `all` and never a list. It is training-time only: the value lands in the checkpoint's `args.json`, generation has no such flag and reads it back from there (to sample another group, sample that group's checkpoint), and a resume may not change it. The group and the clip's free-text `action_label` are maintained by hand in `action_labels.jsonl` (one `{"clip": "<name>.npy", "action_group": "...", "action_label": "..."}` object per line) alongside `motion_metadata.json`.
* Add `--action_label_cond` to condition on the label text. It needs the frozen-T5 sidecar next to the labels — build it once with `python tools/build_action_label_embeddings.py <dataset_dir>` (rebuild after editing `action_labels.jsonl`). `--action_label_coarse_prob 0.3` additionally trains on the coarse string synthesized from the label's controlled words (`"idle, roar"`), so the model answers the short queries users actually type.
* **Recommended:** Add `--use_ema` for Exponential Moving Average to improve performance.
* Use `--diffusion_steps 50` to train the faster model with less diffusion steps.
* Use `--device` to define GPU id.
* Add `--train_platform_type {WandBPlatform, TensorboardPlatform}` to track results with either [WandB](https://wandb.ai/site/) or [Tensorboard](https://www.tensorflow.org/tensorboard).

Example of training only on locomotion clips:
```shell
python -m train.train_anytop --model_prefix all_locomotion --objects_subset all --action_group locomotion --lambda_geo 1.0 --auto_resume --balanced
```

## Visualizing Motions in Blender## Visualizing Motions in Blender
We provide a script to visualize motion data as animated skeletons in Blender, similar to the figures shown in the paper.
The script accepts either a single .bvh file or a directory containing multiple .bvh files. For each file, it generates a corresponding .blend file with the skeleton animation rendered in an empty scene.
**Note**: To export an .mp4 render, you'll need to manually set up lighting and position the camera in the Blender UI.  
Before running the command below, make sure to install the following dependencies in Blender’s Python environment:
* BVH parsing (bundled in codebase, originally from https://github.com/inbar-2344/Motion)
* tqdm
* scipy

To create visualizations for all .bvh files in assets/Truebones_Chicken, run:
```shell
blender -b -P visualization/bvh2skeleton.py -- --bvh_path assets/Truebones_Chicken --save_dir save/blend_files --subset bipeds
```
* **Customization** You can adjust the visualization using --sphere_radius (joint sphere radius), --cylinder_radius (bone cylinder radius), and --scale (global scale factor).

## Acknowledgments
We want to thank the following contributors that our code is based on:
[mdm](https://github.com/GuyTevet/motion-diffusion-model), [GRPE](https://github.com/lenscloth/GRPE/tree/master), [audiocraft](https://github.com/facebookresearch/audiocraft)

## License
This code is distributed under an [MIT LICENSE](LICENSE).
Note that our code depends on other libraries that have their own respective licenses that must also be followed.
