import json
import re
from typing import Any, Dict, List, Optional, Type

import torch
from pydantic import BaseModel, ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer

from gauss_steer.utils.utils import get_model_path


# --- Pydantic Model Definition ---
class NumericalRange(BaseModel):
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    relative_lower_bound: Optional[float] = None
    relative_upper_bound: Optional[float] = None


# --- Mock Response Structures (kept for compatibility) ---
class MockMessage:
    def __init__(self, content: str):
        self.content = content


class MockParsedMessage:
    def __init__(self, parsed_obj: BaseModel):
        self.parsed = parsed_obj


class MockChoice:
    def __init__(self, message: Any):
        self.message = message


class MockCompletion:
    def __init__(self, choices: List[MockChoice]):
        self.choices = choices


# --- Hugging Face Client Implementation ---
class HuggingFaceCompletions:
    """
    Synchronous text completions with true micro-batching.
    API shape mirrors OpenAI's `client.chat.completions` object.
    """

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _generate_batch_sync(
        self,
        batch_messages: List[List[Dict[str, str]]],
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> List[str]:
        """
        Real batched generation. All blocking, fully synchronous.
        """
        if len(batch_messages) == 0:
            return []

        prompts = [self._apply_chat_template(m) for m in batch_messages]

        # Tokenize with padding for batching
        enc = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        enc = {k: v.to(self._model.device) for k, v in enc.items()}

        with torch.no_grad():
            generated = self._model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # LEFT-PADDING CUTOFF: new tokens start at the padded width
        if getattr(self._tokenizer, "padding_side", "right") == "left":
            cutoff = enc["input_ids"].shape[1]  # same for every row in the batch
            responses: List[str] = []
            for i in range(generated.shape[0]):
                resp_ids = generated[i, cutoff:]
                text = self._tokenizer.decode(resp_ids, skip_special_tokens=True)
                responses.append(text.lstrip())
            return responses

        # Fallback if not left-padded (keeps your old behavior)
        responses: List[str] = []
        input_lengths = (enc["input_ids"] != self._tokenizer.pad_token_id).sum(dim=1)
        for i in range(generated.shape[0]):
            start = input_lengths[i].item()
            resp_ids = generated[i, start:]
            text = self._tokenizer.decode(resp_ids, skip_special_tokens=True)
            responses.append(text.lstrip())

        return responses

    # --- Public sync APIs ---
    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        **kwargs,
    ) -> MockCompletion:
        text = self._generate_batch_sync([messages], max_new_tokens, temperature)[0]
        return MockCompletion([MockChoice(MockMessage(text))])

    def batch_create(
        self,
        model: str,
        batch_messages: List[List[Dict[str, str]]],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        **kwargs,
    ) -> List[MockCompletion]:
        texts = self._generate_batch_sync(batch_messages, max_new_tokens, temperature)
        return [MockCompletion([MockChoice(MockMessage(t))]) for t in texts]


class HuggingFaceBetaCompletions:
    """
    Structured responses (Pydantic parsing) with micro-batching.
    Mirrors `client.beta.chat.completions` object.
    """

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self._standard = HuggingFaceCompletions(model, tokenizer)

    def _parse_json_from_text(
        self, text: str, pydantic_model: Type[BaseModel]
    ) -> BaseModel:
        try:
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if not match:
                return pydantic_model()
            json_str = match.group(0)
            data = json.loads(json_str)
            return pydantic_model(**data)
        except json.JSONDecodeError:
            return pydantic_model()
        except ValidationError:
            return pydantic_model()
        except Exception:
            return pydantic_model()

    # --- Public sync APIs ---
    def parse(
        self,
        model: str,
        messages: List[Dict[str, str]],
        response_format: Type[BaseModel],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        **kwargs,
    ) -> MockCompletion:
        completion = self._standard.create(
            model=model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        text = completion.choices[0].message.content
        parsed = self._parse_json_from_text(text, response_format)
        return MockCompletion([MockChoice(MockParsedMessage(parsed))])

    def batch_parse(
        self,
        model: str,
        batch_messages: List[List[Dict[str, str]]],
        response_format: Type[BaseModel],
        max_new_tokens: int = 1024,
        temperature: float = 1.0,
        **kwargs,
    ) -> List[MockCompletion]:
        completions = self._standard.batch_create(
            model=model,
            batch_messages=batch_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        results: List[MockCompletion] = []
        for c in completions:
            text = c.choices[0].message.content
            parsed = self._parse_json_from_text(text, response_format)
            results.append(MockCompletion([MockChoice(MockParsedMessage(parsed))]))
        return results


class HuggingFaceBeta:
    def __init__(self, model, tokenizer):
        self.completions = HuggingFaceBetaCompletions(model, tokenizer)


class HuggingFaceChat:
    def __init__(self, model, tokenizer):
        self.completions = HuggingFaceCompletions(model, tokenizer)


class EvaluationModelClient:
    """
    Synchronous Hugging Face mapper with micro-batching.
    """

    def __init__(self, model_name: str):
        model_name = get_model_path(model_name)
        print(f"Loading model: {model_name}...")
        self.model_name = model_name

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        # Ensure a pad token exists for batched padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # IMPORTANT for decoder-only LMs: use left padding
        self.tokenizer.padding_side = "left"
        # Make sure the model knows its pad token id
        try:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        except Exception:
            pass
        print("Model loaded successfully.")

        self.chat = HuggingFaceChat(self.model, self.tokenizer)
        self.beta = HuggingFaceBeta(self.model, self.tokenizer)
