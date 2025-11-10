"""
This script runs the Gaussian depth schedule experiments for each of the seven models.
The experiments are done on the validation set (25% of MASK benchmark) to the find best hyperparameters.
"""

from gauss_steer.control.train_control_vector import train_control_vector
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

# Range of peak coefficients and std dev factors to evaluate
PEAK_COEFF_RANGE = (0.1, 3.0, 0.1)  # min_range, max_range, step_size
STD_DEV_FACTOR_RANGE = (0.0, 1.0, 0.5)  # min_range, max_range, step_size
VALIDATION = True


def create_run_identifier(config: dict) -> str:
    """
    Create a unique identifier for the experiment.
    """
    return f"{config['model_name']}_schedule=gaussian_peak_peak_coeff={config['peak_coeff']}_std_dev={config['std_dev_factor']}_validation={VALIDATION}"


def frange(min_value: float, max_value: float, step: float):
    """Helper function
    Generate a range of values between min_value and max_value with the given step size.
    """
    return list(range(min_value, max_value + step, step))


def main():
    """Main function to run the experiment."""
    # Generate a range of peak coefficients and std dev factors
    peak_coeffs = list(frange(*PEAK_COEFF_RANGE))
    std_dev_factors = list(frange(*STD_DEV_FACTOR_RANGE))

    for model in MODELS:
        model_name = model.split("/")[-1]

        # Train control vector - save to CONTROL_VECTOR_PATH/<model_name>.pkl
        train_control_vector(model_name)

        # Loop through all peak coefficients and std dev factors
        for peak_coeff in peak_coeffs:
            for std_dev_factor in std_dev_factors:
                # Create config
                control_config = {
                    "schedule": "gaussian",
                    "peak_coeff": peak_coeff,
                    "std_dev_factor": std_dev_factor,
                }
                config = BASE_CONFIG.copy()
                config["model_name"] = model_name
                config["control_config"] = control_config

                # Run MASK pipeline
                identifier = create_run_identifier(config)
                output_folder = "gaussian_depth_schedule" / identifier
                pipeline(
                    config,
                    identifier=identifier,
                    output_folder=output_folder,
                    validation=VALIDATION,
                )


if __name__ == "__main__":
    main()
