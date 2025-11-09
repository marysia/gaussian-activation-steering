from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gauss_steer.utils.constants import MODEL_BASE_PATH


def get_model_path(model_name):
    """
    Returns the path to the model directory for a given model name.

    Args:
        model_name (str): The name of the model to search for.
        base_dir (str, optional): The base directory where models are stored. Defaults to "/data/huggingface".

    Returns:
        str or None: The path to the most recent snapshot directory for the model, or None if not found.

    Raises:
        FileNotFoundError: If the base directory does not exist or is not a directory.
    """
    base_path = Path(MODEL_BASE_PATH)
    if not base_path.exists() or not base_path.is_dir():
        raise FileNotFoundError(
            f"Base directory '{MODEL_BASE_PATH}' does not exist or is not a directory."
        )

    for d in base_path.iterdir():
        if d.is_dir() and d.name.endswith(model_name):
            # Found the model directory, now look for snapshots
            snapshots_dir = d / "snapshots"
            if snapshots_dir.exists() and snapshots_dir.is_dir():
                # Get all subdirectories in snapshots
                snapshot_folders = [f for f in snapshots_dir.iterdir() if f.is_dir()]
                if snapshot_folders:
                    # Return the most recent one (by modification time)
                    most_recent = max(snapshot_folders, key=lambda x: x.stat().st_mtime)
                    return str(most_recent)
    print(f"Model {model_name} not found!")
    return None


def get_model_and_tokenizer(
    model_name: str,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the model and tokenizer for a given model name.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.

    Raises:
        FileNotFoundError: If the model path cannot be found.
        OSError: If loading the model or tokenizer fails.
    """
    model_path = get_model_path(model_name)

    # Get tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # TODO imporve this line
    # # Get model.
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype="auto", device_map="auto"
    )

    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
