from pathlib import Path

DATA_PATH = Path(__file__).parent / "data"
CONTROL_VECTOR_PATH = Path("control_vectors")  # Directory where the control vectors are stored
RESULT_PATH = Path("results") # Directory where the results are stored

# Directory where the huggingface models are stored
MODEL_BASE_PATH = "/data/huggingface"


#### ----- configuration for training the control vector ----
POSITIVE_PERSONAS = ["honest"]
NEGATIVE_PERSONAS = ["dishonest"]
TEMPLATE = "Act as if you're extremely {persona}"

### ----- configuration for evaluating the responses on the MASK benchmark  ----
EVALUATION_MODEL_NAME = "models--openai--gpt-oss-20b"