from pathlib import Path

BASE_PATH = Path("/data/mars")
DATA_PATH = Path("/data/marysia_winkels/projects/gaussian-activation-steering/data")
CONTROL_VECTOR_PATH = Path(
    "/data/marysia_winkels/test_repo_dir/control_vectors"
)  # Directory where the control vectors are stored


# Directory where the huggingface models are stored
MODEL_BASE_PATH = "/data/huggingface"


#### ----- configuration for training the control vector ----
POSITIVE_PERSONAS = ["honest"]
NEGATIVE_PERSONAS = ["dishonest"]
TEMPLATE = "Act as if you're extremely {persona}"
