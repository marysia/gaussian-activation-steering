import argparse
import json

import dill as pickle

from gauss_steer.control.extract import ControlVector, DatasetEntry
from gauss_steer.utils.constants import (
    CONTROL_VECTOR_PATH,
    DATA_PATH,
    NEGATIVE_PERSONAS,
    POSITIVE_PERSONAS,
    TEMPLATE,
)
from gauss_steer.utils.prompt_templates import get_prompt_template_func
from gauss_steer.utils.utils import get_model_and_tokenizer


def get_output_suffixes(tokenizer):
    """
    Generate a list of output suffixes based on truncated output sequences.

    This function loads a JSON file containing truncated outputs, tokenizes each output using the given
    tokenizer, and generates all possible non-empty suffixes by joining 1 to len(tokens)-1 tokens for each output.
    Returns a list of these output suffixes as strings.

    Args:
        tokenizer: The tokenizer to use for tokenizing the outputs.

    Returns:
        list[str]: A list containing all possible non-empty output suffixes.
    """
    with open(DATA_PATH / "control_vector" / "all_truncated_outputs.json") as f:
        output_suffixes = json.load(f)
    return [
        tokenizer.convert_tokens_to_string(tokens[:i])
        for tokens in (tokenizer.tokenize(s) for s in output_suffixes)
        for i in range(1, len(tokens))
    ]


def make_dataset(
    prompt_template_func: callable,
    tokenizer,
    suffix_list: list[str],
) -> list[DatasetEntry]:
    """
    This function constructs a dataset of DatasetEntry objects, each representing a pair of positive and negative prompts
    generated using a prompt template and different personas, by attaching all given suffixes to these templates.

    Args:
        prompt_template_func: callable, the function to use to generate the prompt
        tokenizer: the tokenizer to use to tokenize the prompts
        suffix_list: list[str], the list of suffixes to attach to the prompts

    Returns:
        list[DatasetEntry], the dataset of DatasetEntry objects
    """
    dataset = []
    for suffix in suffix_list:
        for positive_persona, negative_persona in zip(
            POSITIVE_PERSONAS, NEGATIVE_PERSONAS
        ):
            # Generate positive and negative templates for each suffix
            positive_template = TEMPLATE.format(persona=positive_persona)
            negative_template = TEMPLATE.format(persona=negative_persona)
            dataset.append(
                DatasetEntry(
                    positive=prompt_template_func(positive_template, tokenizer)
                    + suffix,
                    negative=prompt_template_func(negative_template, tokenizer)
                    + suffix,
                )
            )
    return dataset


def train_control_vector(model_name):
    """
    Trains a control vector for a given model.

    This function prepares the dataset and sets up the environment to train a control vector
    (e.g., a direction in latent space) based on positive/negative prompt pairs. The trained vector
    can be used for controlled generation or analysis.

    Args:
        model_name (str): The name or identifier of the model to use for training.

    Process:
        1. Retrieves the prompt template function for the given model.
        2. Loads the model and tokenizer.
        3. Reads output suffixes from disk and prepares a dataset of positive/negative prompt examples.
        4. (Commented out) Optionally trains a control vector using the prepared dataset.
        5. (Commented out) Optionally saves the trained control vector to disk.
    """
    # Retrieve a prompt template function tailored to this model
    prompt_template_func = get_prompt_template_func(model_name)

    # Load the model and tokenizer based on the model name
    model, tokenizer = get_model_and_tokenizer(model_name)

    # Obtain all suffixes to append to the prompts—these are phrases/tokens to steer generation
    suffixes = get_output_suffixes(tokenizer)

    # Build a list of DatasetEntry objects: each containing a positive and negative sample string
    dataset = make_dataset(prompt_template_func, tokenizer, suffixes)

    # Train the control vector using the constructed dataset, model, and tokenizer
    control_vector = ControlVector.train(model, tokenizer, dataset)

    # Ensure the target directory exists before attempting to write the control vector
    CONTROL_VECTOR_PATH.mkdir(parents=True, exist_ok=True)

    # Save the trained control vector as a pickle file for future use
    with open(CONTROL_VECTOR_PATH / f"{model_name}.pkl", "wb") as f:
        pickle.dump(control_vector, f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a control vector for the specified model."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Name or identifier of the model to use for training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_control_vector(args.model_name)


if __name__ == "__main__":
    main()
