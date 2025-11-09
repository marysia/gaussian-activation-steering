# Gaussian Activation Steering


## Training a Control Vector

This repository provides a script to train a "control vector" for language models, which can be used to steer model activations in a desired direction (for instance, making output more "honest" or "dishonest").

The script works by loading positive and negative prompt examples, extracting their activations from a specified model, and computing directions using these pairs. The trained control vector can then be saved and applied for further experiments or generation control.

### How to Use

The script can be invoked via the command line using:

```bash
train-control-vector --model-name <MODEL_NAME>
```

Replace `<MODEL_NAME>` with the exact name or identifier of your HuggingFace model (for example, `Llama-3.2-1B-Instruct`). This assumes the HuggingFace model is saved locally in `<MODEL_BASE_PATH>`, defined in `gauss_steer.utils.constants.py`.

This command will:
- Generate pairs of positive and negative prompts based on predefined templates and personas.
- Extract model activations for these prompts.
- Train a control vector that represents the "direction" between them.
- Save the resulting control vector as a pickle file.

**Example:**
```bash
train-control-vector --model-name Llama-3.2-1B-Instruct
```
You can find this command registered as an entry point in `pyproject.toml` under `[project.scripts]`.

For more details on customizing prompts, personas, or dataset locations, see the relevant configuration constants in the codebase under `gauss_steer/utils/constants.py`.

## MASK benchmark 
The `MASK` directory contains code and utilities for running the MASK benchmark, which is designed to evaluate language models on honest and dishonest reasoning using masked statements.

### Contents of the `MASK` Directory

- `generate.py`: Generates responses to MASK benchmark prompts using a given model.
- `evaluate.py`: Evaluates the responses by scoring and analyzing them.
- `calculate.py`: Computes various metrics from the evaluated results.
- `aggregate.py`: Aggregates final metrics for easy comparison.
- `pipeline.py`: Provides a single script for running the full MASK benchmark pipeline.
- `data/` (not versioned here): Contains the raw CSVs and datasets used for the MASK benchmark.

### Running the MASK Benchmark Pipeline

You can run the entire MASK benchmark workflow with the following command-line interface provided by the `mask-pipeline` entry point:

```bash
mask-pipeline --config-path <CONFIG_PATH> --output-folder <OUTPUT_FOLDER> --identifier <IDENTIFIER>
```

- `<CONFIG_PATH>`: Path to a .yaml file that contains the details of the model. This can be a baseline model (e.g., `Llama-3.2-1B-Instruct`) or a baseline model with a control vector applied to it, in which case `control_vector_path` and `control_config` need to be set. 
- `<OUTPUT_FOLDER>`: Subdirectory inside the MASK results folder where outputs will be written (acts as a run label or group name)
- `<IDENTIFIER>`: A unique string to distinguish this run's result files (for reproducibility or batch processing)

**Example Usage:**
```bash
mask-pipeline --model-name configs/example-steering-config.yaml --output-folder llama-1B --identifier gaussian_steering
```

This pipeline will:
1. Generate model responses to all MASK benchmark CSVs in the data directory.
2. Evaluate the truthfulness and consistency of those responses.
3. Calculate and aggregate quantitative metrics.
4. Save all intermediate and final outputs to organized subfolders under your specified `<OUTPUT_FOLDER>` (inside `RESULT_PATH/mask/`).

For more details on paths or configuration, refer to the constants in `gauss_steer/utils/constants.py`.


## LoRRA
The LoRRA (LoRA with Representation Alignment) framework allows you to train or fine-tune language models with custom alignment or control objectives. This was implemented to provide a contrast with the post-hoc approach of (Gaussian) activation steering. 

### Training with LoRRA
The `train-lorra` script (registered as an entry point in `pyproject.toml`) provides a CLI for launching LoRRA training. It requires a YAML configuration file specifying model, data, and hyperparameter options.

**Example command:**
```bash
train-lorra --config-path <CONFIG_PATH> --output-path <OUTPUT_PATH>
```
To change personas, templates, or data, edit the entries in your config and/or adjust the constants in `gauss_steer/utils/constants.py` as needed.

