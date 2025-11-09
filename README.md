To be done.


## Training a Control Vector

This repository provides a script to train a "control vector" for language models, which can be used to steer model activations in a desired direction (for instance, making output more "honest" or "dishonest").

The script works by loading positive and negative prompt examples, extracting their activations from a specified model, and computing directions using these pairs. The trained control vector can then be saved and applied for further experiments or generation control.

### How to Use

The script can be invoked via the command line using the entry point defined in `pyproject.toml`:

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

### Arguments

- `--model-name`: (required) Name or identifier of the model to use for training.

For more details on customizing prompts, personas, or dataset locations, see the relevant configuration constants in the codebase under `gauss_steer/utils/constants.py`.
