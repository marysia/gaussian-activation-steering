import argparse
from pathlib import Path

from gauss_steer.mask.generate import generate_responses
from gauss_steer.mask.evaluate import evaluate_responses
from gauss_steer.mask.calculate import calculate_metrics
from gauss_steer.mask.aggregate import aggregate_metrics
from gauss_steer.utils.model import HuggingFaceModelClient


def pipeline(
    config: str | Path,
    output_folder: str | Path,
    identifier: str,
    validation: bool = False,
) -> None:
    """
    Pipeline for the MASK benchmark.
    """
    output_path = Path(output_folder)
    output_folder = output_path.as_posix()

    client = HuggingFaceModelClient.load_from_config(config)
    generate_responses(client, output_folder, identifier, validation=validation)
    evaluate_responses(output_folder)
    calculate_metrics(output_folder)
    aggregate_metrics(output_folder)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MASK benchmark pipeline.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config describing the generation model.",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Subdirectory inside RESULT_PATH/mask/ where outputs are written.",
    )
    parser.add_argument(
        "--identifier",
        required=True,
        help="Identifier appended to filenames to distinguish runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pipeline(
        config_path=Path(args.config_path),
        output_folder=Path(args.output_folder),
        identifier=args.identifier,
    )


if __name__ == "__main__":
    main()