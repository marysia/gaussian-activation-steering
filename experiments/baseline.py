"""
This script runs the MASK benchmark for each of the seven models.
This means no adjustments to the model are made, so it is a baseline for the other experiments.
"""

from gauss_steer.mask.pipeline import pipeline

# All models to be evaluated
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]

# Base config for all models. model_name and control_config will be overwritten for each model and control config.
BASE_CONFIG = {
    "model_name": None,
    "temperature": 0.01,
    "max_tokens": 1024,
    "output_format_chat": False,
    "top_p": None,
    "top_k": None,
    "repetition_penalty": None,
}

VALIDATION = True


def main():
    """Main function to run the experiment."""
    for model in MODELS:
        model_name = model.split("/")[-1]

        config = BASE_CONFIG.copy()
        config["model_name"] = model_name

        # Run MASK pipeline
        identifier = model_name
        output_folder = "baseline"
        pipeline(
            config,
            identifier=identifier,
            output_folder=output_folder,
            validation=VALIDATION,
        )


if __name__ == "__main__":
    main()
