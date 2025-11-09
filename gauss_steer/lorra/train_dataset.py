from torch.utils.data import Dataset
from datasets import load_dataset
import transformers
from typing import Dict
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd

from gauss_steer.utils.constants import TEMPLATE, POSITIVE_PERSONAS, NEGATIVE_PERSONAS

max_res_len = 64


def get_assistant_token(tokenizer): 
    if '<|start_header_id|>system<|end_header_id|>\\n\\n' in tokenizer.chat_template:
        return '<|start_header_id|>assistant<|end_header_id|>\n\n'   # NOTE! \n not escaped here. 

    elif '<|im_start|>assistant\\n' in tokenizer.chat_template:
        return '<|im_start|>assistant\n'   # NOTE! \n not escaped here. 

    elif '[/INST]' in tokenizer.chat_template:
        return "[/INST]"

    raise ValueError('Help! No assistant token!')


class AlpacaSupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                tokenizer: transformers.PreTrainedTokenizer, 
                n_samples,
                lorra_args,
                ):
        super(Dataset, self).__init__()

        self.n_samples = n_samples
        self.max_res_len = lorra_args.max_res_len

        self.tokenizer = tokenizer
        if self.tokenizer.chat_template is None: 
            raise ValueError('Tokenizer does not have chat template. This is not supported.')

        self._set_chat_messages()


    def _set_chat_messages(self):
        # Load the dataset. 
        ds = load_dataset('tatsu-lab/alpaca')
        ds = ds.filter(lambda x: x['input'] == '')
        instructions = ds['train']['instruction']
        responses = ds['train']['output']

        # Create positive and negative examples.
        self.positive_samples = []
        self.negative_samples = [] 
        for instruction, response in zip(instructions, responses): 
            
            prompt = f'{instruction} {TEMPLATE} {POSITIVE_PERSONAS[0]}'
            chat = [
                [{'role': 'user', 'content': prompt}, 
                {'role': 'assistant', 'content': response}]
            ]
            self.positive_samples.extend(self.tokenizer.apply_chat_template(chat, tokenize=False))

            prompt = f'{instruction} {TEMPLATE} {NEGATIVE_PERSONAS[0]}'
            chat = [
                [{'role': 'user', 'content': prompt}, 
                {'role': 'assistant', 'content': response}]
            ]
            self.negative_samples.extend(self.tokenizer.apply_chat_template(chat, tokenize=False))

            if len(self.positive_samples) > self.n_samples:
                break

        assert len(self.positive_samples) == len(self.negative_samples)

    def __len__(self):
        return len(self.positive_samples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assistant_tag = get_assistant_token(self.tokenizer)

        positive_sample = self.positive_samples[i]
        positive_input, positive_output = positive_sample.split(assistant_tag)
        assert len(positive_sample) > len(positive_input)
        assert len(positive_sample) > len(positive_output)

        negative_sample = self.negative_samples[i]
        negative_input, negative_output = negative_sample.split(assistant_tag)
        assert len(negative_sample) > len(negative_input)
        assert len(negative_sample) > len(negative_output)

        # Tokenize imputs
        self.tokenizer.padding_side = "left"
        tokenized_inputs = self.tokenizer([positive_input, negative_input],
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        # Tokenize outputs
        self.tokenizer.padding_side = "right"
        response_tokenized_inputs = self.tokenizer([
            assistant_tag + positive_output, 
            assistant_tag + negative_output],
            padding="max_length",
            truncation=True,
            max_length=self.max_res_len,
            return_tensors="pt",
        )
        
        # Combine input and outputs for ids and mask. 
        combined_input_ids = torch.cat([tokenized_inputs["input_ids"], response_tokenized_inputs["input_ids"]], dim=1)
        combined_attention_mask = torch.cat([tokenized_inputs["attention_mask"], response_tokenized_inputs["attention_mask"]], dim=1)
        return dict(
            input_ids=combined_input_ids,
            attention_mask=combined_attention_mask
        )
