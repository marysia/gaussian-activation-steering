from typing import Callable

from transformers import AutoTokenizer


def get_prompt_template_func(model_name: str) -> Callable[[str, AutoTokenizer], str]:
    """
    Return the appropriate prompt template function based on the model name.

    arguments:
        model_name: str, the name of the model

    returns: Callable[[str, any], str], the prompt template function for the model
    """

    if "llama" in model_name.lower():
        return build_prompt_llama3
    elif "mistral" in model_name.lower():
        return build_prompt_mistral
    elif "qwen" in model_name.lower():
        return build_prompt_qwen3
    elif "gpt" in model_name.lower():
        return build_prompt_gpt
    else:
        raise ValueError(f"Model {model_name} not supported")


def build_prompt_qwen3(user_message: str, tokenizer: AutoTokenizer) -> str:
    """
    Return a single-string prompt for a Qwen3-30B-A3B-Instruct-2507 model
    where the assistant should begin generating right after the header.

    arguments:
        user_message: str, the message from the user
        tokenizer: the tokenizer to use for applying the chat template

    returns: str, the prompt formatted for the model
    """

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_message},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_prompt_llama3(prompt: str, tokenizer: AutoTokenizer) -> str:
    """
    Return a single-string prompt for a Llama-3.1-Instruct model
    where the assistant should begin generating right after the header.
    """
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_prompt_mistral(prompt: str, tokenizer: AutoTokenizer) -> str:
    """
    Return a single-string prompt for a Mistral-7B-Instruct model
    where the assistant should begin generating right after the header.
    """
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_prompt_gpt(prompt: str, tokenizer: AutoTokenizer) -> str:
    """
    Return a single-string prompt for a GPT model
    where the assistant should begin generating right after the header.
    """
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": prompt}]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
