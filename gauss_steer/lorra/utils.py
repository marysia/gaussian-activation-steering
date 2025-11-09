import torch
import gc
from transformers import Trainer

def compute_loss(model, inputs, target_layers, alpha, beta, max_res_len=64, return_outputs=False, **kwargs):

    input_ids = inputs.get("input_ids")
    attention_mask = inputs.get("attention_mask")

    assert input_ids.shape[1] == 2

    pos_input_ids = input_ids[:, 0]
    neg_input_ids = input_ids[:, 1]

    pos_attention_mask = attention_mask[:, 0]
    neg_attention_mask = attention_mask[:, 1]

    min_length = max_res_len
    response_attention_mask = pos_attention_mask[:, -min_length:].repeat(len(target_layers), 1, 1).unsqueeze(-1)

    with model.disable_adapter():
        model.eval()
        with torch.no_grad():
            pos_outputs = model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            pos_hidden = [pos_outputs[l][:, -min_length:].detach() for l in target_layers]

            neg_outputs = model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                output_hidden_states=True
            )['hidden_states']
            direction_hidden = [pos_outputs[l][:, -min_length:].detach() - neg_outputs[l][:, -min_length:].detach() for l in target_layers]
            target_hidden = torch.stack([pos_hidden[i] + alpha * direction_hidden[i] for i in range(len(target_layers))]) * response_attention_mask

            del pos_outputs, neg_outputs, pos_hidden, direction_hidden
            gc.collect()
            torch.cuda.empty_cache()

    model.train()
    lora_outputs = model(
        input_ids=pos_input_ids,
        attention_mask=pos_attention_mask,
        output_hidden_states=True
    )['hidden_states']
    lora_hidden = torch.stack([lora_outputs[l][:, -min_length:] for l in target_layers]) * response_attention_mask

    loss = torch.norm(lora_hidden - target_hidden, dim=-1, p=2, dtype=torch.float).nanmean()
    return (loss, lora_hidden) if return_outputs else loss

class CustomTrainer(Trainer):
    def __init__(self, lorra_target_layers, alpha, beta, max_res_len, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the parent class's __init__
        self.alpha = alpha
        self.beta = beta
        self.max_res_len = max_res_len
        self.lorra_target_layers = lorra_target_layers

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        return compute_loss(
            model, 
            inputs,
            target_layers=self.lorra_target_layers,
            alpha=self.alpha,
            beta=self.beta,
            max_res_len=self.max_res_len,
            return_outputs=return_outputs,
            **kwargs
        )
