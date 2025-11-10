"""
This script runs the ablation experiments for the Gaussian depth schedule.
The experiments are:
1. Gaussian random
2. Uniform Gaussian
3. Box filter on given starting layer

Experiments are done on the validation set (25% of MASK benchmark) to the find best hyperparameters.
"""

from gauss_steer.control.train_control_vector import train_control_vector
from gauss_steer.mask.pipeline import pipeline

# All models to be evaluated
MODELS = ["meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct"]

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


def frange(min_value: float, max_value: float, step: float):
    """Helper function
    Generate a range of values between min_value and max_value with the given step size.
    """
    return list(range(min_value, max_value + step, step))


def ablation_gaussian_random():
    peak_coeff_range = (0.1, 3.0, 0.1)  # min_range, max_range, step_size
    std_dev_factor_range = (0.0, 1.0, 0.5)  # min_range, max_range, step_size

    peak_coeffs = list(frange(*peak_coeff_range))
    std_dev_factors = list(frange(*std_dev_factor_range))

    for model in MODELS:
        model_name = model.split("/")[-1]

        # Train control vector - save to CONTROL_VECTOR_PATH/<model_name>.pkl
        train_control_vector(model_name)

        for peak_coeff in peak_coeffs:
            for std_dev_factor in std_dev_factors:
                # Create config
                control_config = {
                    "schedule": "gaussian_random",
                    "peak_coeff": peak_coeff,
                    "std_dev_factor": std_dev_factor,
                }

                config = BASE_CONFIG.copy()
                config["model_name"] = model_name
                config["control_config"] = control_config

                # Run MASK pipeline
                identifier = f"{config['model_name']}_schedule=gaussian_random_peak_coeff={config['peak_coeff']}_std_dev={config['std_dev_factor']}_validation={VALIDATION}"
                output_folder = "ablation_gaussian_random" / identifier
                pipeline(
                    config,
                    identifier=identifier,
                    output_folder=output_folder,
                    validation=VALIDATION,
                )


def ablation_uniform_gaussian():
    peak_coeff_range = (0.1, 3.0, 0.1)  # min_range, max_range, step_size
    std_dev_factor_range = (0.0, 1.0, 0.5)  # min_range, max_range, step_size

    peak_coeffs = list(frange(*peak_coeff_range))
    std_dev_factors = list(frange(*std_dev_factor_range))

    for model in MODELS:
        model_name = model.split("/")[-1]

        # Train control vector - save to CONTROL_VECTOR_PATH/<model_name>.pkl
        train_control_vector(model_name)

        for peak_coeff in peak_coeffs:
            for std_dev_factor in std_dev_factors:
                # Create config
                control_config = {
                    "schedule": "uniform_gaussian",
                    "peak_coeff": peak_coeff,
                    "std_dev_factor": std_dev_factor,
                }

                config = BASE_CONFIG.copy()
                config["model_name"] = model_name
                config["control_config"] = control_config

                # Run MASK pipeline
                identifier = f"{config['model_name']}_schedule=uniform_gaussian_peak_coeff={config['peak_coeff']}_std_dev={config['std_dev_factor']}_validation={VALIDATION}"
                output_folder = "ablation_uniform_gaussian" / identifier
                pipeline(
                    config,
                    identifier=identifier,
                    output_folder=output_folder,
                    validation=VALIDATION,
                )


def ablation_box_filter():
    n_layers = 32  # Llama 8B and Qwen 7B both have 32 layers.

    start_layer_range = (10, n_layers - 2, 1)  # min_range, max_range, step_size
    k_layers_range = (2, 26, 2)  # min_range, max_range, step_size
    peak_coeff_range = (0.1, 3.0, 0.1)  # min_range, max_range, step_size
    std_dev_factor_range = (0.0, 1.0, 0.5)  # min_range, max_range, step_size

    peak_coeffs = list(frange(*peak_coeff_range))
    std_dev_factors = list(frange(*std_dev_factor_range))
    start_layers = list(frange(*start_layer_range))
    k_layers = list(frange(*k_layers_range))

    for model in MODELS:
        model_name = model.split("/")[-1]

        # Train control vector - save to CONTROL_VECTOR_PATH/<model_name>.pkl
        train_control_vector(model_name)

        for start_layer in start_layers:
            for k_layer in k_layers:
                for peak_coeff in peak_coeffs:
                    for std_dev_factor in std_dev_factors:
                        # Create config
                        control_config = {
                            "schedule": "box_filter_on_given_starting_layer",
                            "peak_coeff": peak_coeff,
                            "std_dev_factor": std_dev_factor,
                            "start_layer": start_layer,
                            "k_layers": k_layer,
                        }

                        config = BASE_CONFIG.copy()

                    config = BASE_CONFIG.copy()
                    config["model_name"] = model_name
                    config["control_config"] = control_config

                    # Run MASK pipeline
                    identifier = f"{config['model_name']}_schedule=box_filter_peak_coeff={config['peak_coeff']}_std_dev={config['std_dev_factor']}_validation={VALIDATION}"
                    output_folder = "ablation_box_filter" / identifier
                    pipeline(
                        config,
                        identifier=identifier,
                        output_folder=output_folder,
                        validation=VALIDATION,
                    )


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
