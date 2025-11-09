import argparse
from pathlib import Path
from typing import Tuple

import yaml
from peft import LoraConfig, get_peft_model
import transformers


from gauss_steer.lorra.train_dataset import AlpacaSupervisedDataset
from gauss_steer.lorra.argument_dataclasses import ModelArguments, TrainingArguments, LoraArguments, LorraArguments
from gauss_steer.lorra.utils import CustomTrainer


def parse_cli_arguments() -> Tuple[Path, Path]:
    parser = argparse.ArgumentParser(description="Train LoRRA model using a YAML configuration file.")
    parser.add_argument(
        "--config-path",
        dest="config_path",
        required=True,
        help="Path to the YAML file containing model, training, LoRA, and LoRRA configuration.",
    )
    parser.add_argument(
        "--output-path",
        dest="output_path",
        required=True,
        help="Directory where the trained model will be saved.",
    )
    args = parser.parse_args()
    config_path = Path(args.config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    return config_path, output_path


def load_configuration(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def parse_training_arguments(config: dict):
    argument_sections = ("model_args", "training_args", "lora_args", "lorra_args")
    merged_arguments = {}
    for section in argument_sections:
        section_values = config.get(section, {})
        if section_values is None:
            continue
        if not isinstance(section_values, dict):
            raise ValueError(f"Configuration section `{section}` must be a mapping.")
        merged_arguments.update(section_values)

    parser = transformers.HfArgumentParser(
        (ModelArguments, TrainingArguments, LoraArguments, LorraArguments)
    )
    model_args, training_args, lora_args, lorra_args = parser.parse_dict(
        merged_arguments, allow_extra_keys=True
    )
    dataset_config = config.get("dataset", {}) or {}
    return model_args, training_args, lora_args, lorra_args, dataset_config


def get_model(model_args, training_args, lora_config):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        device_map="auto"
    )    
    return get_peft_model(model, lora_config)

def get_tokenizer(model_args, training_args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    if not tokenizer.pad_token:
        if tokenizer.unk_token: 
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token
        else: 
            tokenizer.pad_token = '<unk>'
    return tokenizer 

def get_training_dataset(tokenizer, lorra_args, n=10000):
    return AlpacaSupervisedDataset(tokenizer=tokenizer, n_samples=n, lorra_args=lorra_args)


def train(config_path: Path, output_path: Path):
    config = load_configuration(config_path)
    (
        model_args,
        training_args,
        lora_args,
        lorra_args,
        dataset_config,
    ) = parse_training_arguments(config)
    training_args.output_dir = str(output_path)

    # Set LoRRA target layers.
    lorra_target_layers = [int(layer) for layer in lorra_args.target_layers.split(",")]  # target representations
    
    # Set LoRA config. 
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        layers_to_transform=list(range(lorra_target_layers[-1] + 1)),  # LoRA layers
        task_type="CAUSAL_LM",
    )

    # Get tokenizer, model, train_dataset
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args, lora_config)
    train_dataset = get_training_dataset(
        tokenizer=tokenizer,
        lorra_args=lorra_args,
        n=dataset_config.get("n_samples", 10000),
    )

    # Set gradient checkpointing. 
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()


    # Define trainer. 
    trainer = CustomTrainer(
        lorra_target_layers=lorra_target_layers,
        alpha=lorra_args.lorra_alpha,
        beta=lorra_args.lorra_beta,
        max_res_len=lorra_args.max_res_len,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
    )
    model.config.use_cache = False

    # Train! 
    trainer.train()
    trainer.save_state()

    # Save trained model. 
    if getattr(training_args, "local_rank", -1) in (-1, 0):
        # model.save_pretrained(training_args.output_dir) # saving adapter
        merged_model = model.merge_and_unload() # saving full model
        merged_model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)


def main() -> None:
    config_path, output_path = parse_cli_arguments()
    train(config_path, output_path)


if __name__ == "__main__":
    main()

