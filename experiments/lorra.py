

from gauss_steer.lorra.train_lorra import train
from gauss_steer.mask.pipeline import pipeline

# All models to be evaluated
MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

# Base config for all models. model_name will be overwritten for each model and control config.
BASE_CONFIG = {
    "model_name": None,
    "temperature": 0.01,
    "max_tokens": 1024,
    "output_format_chat": False,
    "top_p": None,
    "top_k": None,
    "repetition_penalty": None,
}



# Training args
learning_rates = [5e-5, 1e-4]
num_train_epochs = [2, 3]
gradient_accumulation_steps = [1, 2]
weight_decay = [0.0, 0.05]
lr_scheduler_type = ["cosine", "linear"]
optim = ["adamw_torch", "paged_adamw_32bit"]

# LoRA args
lora_r = [8, 16]
lora_alpha = [16, 32]
lora_dropout = [0.05, 0.1]
lora_target_modules = [["q_proj", "k_proj", "v_proj", "o_proj"], ["q_proj", "v_proj"]

# LoRRA args
target_layers = [["8,12,16,20"], ["12,16,20,24"]]
lorra_beta = [0.0, 0.05]
dataset_size = [10000, 30000]

def train_lorra_models(model: str): 
    # Train LoRRA model (not loving this for loop structure...)
    for learning_rate in learning_rates:
        for num_train_epochs in num_train_epochs:
            for gradient_accumulation_steps in gradient_accumulation_steps:
                for weight_decay in weight_decay:
                    for lr_scheduler_type in lr_scheduler_type:
                        for optim in optim:
                            for lora_r in lora_r:
                                for lora_alpha in lora_alpha:
                                    for lora_dropout in lora_dropout:
                                        for lora_target_modules in lora_target_modules:
                                            for target_layers in target_layers:
                                                for lorra_beta in lorra_beta:
                                                    for dataset_size in dataset_size:
                                                        # Train LoRRA model
                                                        config = {
                                                            "model_name": model,
                                                            "learning_rate": learning_rate,
                                                            "num_train_epochs": num_train_epochs,
                                                            "gradient_accumulation_steps": gradient_accumulation_steps,
                                                            "weight_decay": weight_decay,
                                                            "lr_scheduler_type": lr_scheduler_type,
                                                            "optim": optim,
                                                            "lora_r": lora_r,
                                                            "lora_alpha": lora_alpha,
                                                            "lora_dropout": lora_dropout,
                                                            "lora_target_modules": lora_target_modules,
                                                            "target_layers": target_layers,
                                                            "lorra_beta": lorra_beta,
                                                            "dataset_size": dataset_size,
                                                        }
                                                        output_path = f"lorra/{model}/learning_rate={learning_rate}_num_train_epochs={num_train_epochs}_gradient_accumulation_steps={gradient_accumulation_steps}_weight_decay={weight_decay}_lr_scheduler_type={lr_scheduler_type}_optim={optim}_lora_r={lora_r}_lora_alpha={lora_alpha}_lora_dropout={lora_dropout}_lora_target_modules={lora_target_modules}_target_layers={target_layers}_lorra_beta={lorra_beta}_dataset_size={dataset_size}"
                                                        train(config, output_path)

def main(): 
    for model in MODELS: 
        # Train all LoRRA models for this model
        train_lorra_models(model)

        # Evaluate all on mask pipeline
        base_path = Path(f'lorra/{model}')
        for model_path in base_path.glob('*'):
            # Run mask pipeline
            config = BASE_CONFIG.copy()
            config['model_name'] = model
            output_folder = f"{model}/{model_path.name}"
            identifier = model_path.name
            pipeline(config, output_folder=output_folder, identifier=identifier, validation=True)

if __name__ == "__main__":
    main()