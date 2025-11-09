from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import torch
import dill as pickle
import yaml

from gauss_steer.control.control import ControlModel
from gauss_steer.utils.utils import get_model_and_tokenizer
from gauss_steer.utils.constants import CONTROL_VECTOR_PATH

# -------------------------------
# Helpers (model identification)
# -------------------------------


def _is_qwen(name: str) -> bool:
    return "qwen" in name.lower()


def _is_llama(name: str) -> bool:
    return "llama" in name.lower()


def _is_mistral_or_mixtral(name: str) -> bool:
    n = name.lower()
    return "mistral" in n or "mixtral" in n


def _is_qwen_thinking_variant(name: str) -> bool:
    # Qwen3-*Thinking-2507 models are "thinking-only"
    return "thinking" in name.lower()


def _normalize_repo_id(model_name: str) -> str:
    """
    Accepts flexible inputs and normalizes to the canonical HF repo id.
    """
    if "_strength=" in model_name:
        model_name = model_name.split("_strength=")[0]

    # If a provider prefix is already present, keep it.
    if "/" in model_name:
        return model_name

    # Add missing org prefixes based on family
    if "Llama" in model_name or "LLaMA" in model_name or "Llama-3" in model_name:
        return f"meta-llama/{model_name}"
    if "Mixtral" in model_name or "Mistral" in model_name:
        # Allow either pretrained or instruct; caller should provide exact name they want
        return f"mistralai/{model_name}"
    if "Qwen" in model_name:
        return f"Qwen/{model_name}"

    # Otherwise pass through unchanged (user may be pointing to a local folder)
    return model_name


# -------------------------------
# Core client
# -------------------------------


class HuggingFaceModelClient:
    """
    Synchronous, micro-batched client for Llama-3, Mixtral/Mistral, and Qwen3 models.
    - Optional steering vectors via ControlModel (same API as before).
    - Robust chat templating (HF `apply_chat_template`) with add_generation_prompt=True.
    - Device-aware, left padding for proper batching.
    - Per-model EOS/stop-id handling (e.g., Llama-3 <|eot_id|>).
    - Optional Qwen 'thinking' support (strip <think>…</think> from final text).
    """

    model: Any
    tokenizer: Any
    model_name: str

    def __init__(
        self,
        model_name: str,
        control_config: Optional[Dict[str, Any]] = None,
        temperature: float = 0.4,
        max_tokens: int = 1024,
        control_vector_path: str = None,
        output_format_chat: bool = False,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.control_config = control_config
        self.original_model_name = model_name
        self.model_name = _normalize_repo_id(model_name)
        self.control_vector_path = control_vector_path

        # Load tokenizer + base model (optionally wrap with control)
        self._set_tokenizer_and_model()

        # Inference defaults
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_format_chat = output_format_chat
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty

        # Ensure sane padding for batched decode-only
        # (left padding avoids shifting positional encodings when padding)
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            # Safe default: use EOS as pad to silence HF warnings
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @classmethod
    def load_from_config(
        cls, config: Union[str, Path, Dict[str, Any]]
    ) -> "HuggingFaceModelClient":
        """
        Construct a HuggingFaceModelClient from a YAML configuration file or dictionary.
        """

        config_path: Optional[Path]
        if isinstance(config, (str, Path)):
            config_path = Path(config)
            with config_path.open("r") as file:
                raw_config = yaml.safe_load(file) or {}
        elif isinstance(config, dict):
            raw_config = dict(config)
            config_path = None
        else:
            raise TypeError(
                "config must be a path-like object or a dictionary of settings."
            )

        def _normalize(value: Any) -> Any:
            if isinstance(value, str) and value.strip().lower() == "none":
                return None
            if isinstance(value, dict):
                return {k: _normalize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_normalize(item) for item in value]
            return value

        config_data = {k: _normalize(v) for k, v in raw_config.items()}

        init_kwargs: Dict[str, Any] = {
            key: config_data.get(key)
            for key in [
                "model_name",
                "control_config",
                "temperature",
                "max_tokens",
                "control_vector_path",
                "output_format_chat",
                "top_p",
                "top_k",
                "repetition_penalty",
            ]
            if key == "model_name" or config_data.get(key) is not None
        }

        if "model_name" not in init_kwargs or not init_kwargs["model_name"]:
            location = (
                f" at {config_path}"
                if config_path is not None
                else ""
            )
            raise ValueError(f"Configuration{location} must specify a 'model_name'.")

        return cls(**init_kwargs)

    # ---------------------------
    # Load model / control vectors
    # ---------------------------

    def _set_tokenizer_and_model(self) -> None:
        model, tokenizer = get_model_and_tokenizer(self.model_name)

        # Optional: wrap with ControlModel if configuration present
        if self.control_config is not None:

            if self.control_vector_path is None:
                self.control_vector_path = f"{CONTROL_VECTOR_PATH}/{self.original_model_name}.pkl"

            # Determine hidden layer range.
            n_layers = len(model.model.layers)
            hidden_layer_range = list(range(1, n_layers))

            # Initiate ControlModel
            control_model = ControlModel(model, hidden_layer_range)
            print("Reading from ", self.control_vector_path)
            control_vector = pickle.load(open(self.control_vector_path, "rb"))
            schedule = self.control_config.get("schedule")

            if schedule == "constant":
                control_model.set_control_constant_scheduler(
                    control_vector, self.control_config["coeff"]
                )

            elif schedule == "box_filter_on_given_starting_layer":
                control_model.set_box_filter_on_given_starting_layer_and_streach_to_l1_gauss_energy(
                    control_vector,
                    self.control_config["peak_coeff"],
                    self.control_config["std_dev_factor"],
                    self.control_config["start_layer"],
                    self.control_config["k_layers"],
                )
            elif schedule == "box_filter":
                control_model.set_box_filter(
                    control_vector,
                    self.control_config["peak_coeff"],
                    std_dev_factor=self.control_config.get("std_dev_factor", 0.25),
                    k_layers=self.control_config.get("k_layers", None),
                    constant=self.control_config.get("constant", 0.0),
                )
            elif schedule == "linear":
                control_model.set_control_with_linear_schedule(
                    control_vector,
                    self.control_config["start_coeff"],
                    self.control_config["end_coeff"],
                )
            elif schedule == "parabolic":
                control_model.set_control_with_parabolic_schedule(
                    control_vector,
                    self.control_config["max_coeff"],
                    self.control_config["min_coeff"],
                )
            elif schedule == "gaussian":
                control_model.set_control_with_gaussian_schedule(
                    control_vector,
                    self.control_config["peak_coeff"],
                    std_dev_factor=self.control_config.get("std_dev_factor", 0.25),
                )
            elif schedule == "sigmoid":
                control_model.set_control_with_sigmoid_schedule(
                    control_vector,
                    self.control_config["max_coeff"],
                    schedule_type=self.control_config.get("schedule_type", "middle"),
                )
            elif schedule == "iterative":
                control_model.set_control_with_iterative_add_one_layer_schedule(
                    control_vector,
                    self.control_config["max_coeff"],
                    self.control_config["step"],
                )
            elif schedule == "energy_gaussian_schedule":
                control_model.set_control_energy_gaussian_schedule(
                    control_vector,
                    self.control_config["tau"],
                    normalize=self.control_config.get("normalize", None),
                )
            elif schedule == "energy_laplace_schedule":
                control_model.set_control_energy_laplace_schedule(
                    control_vector,
                    self.control_config["tau"],
                )
            elif schedule == "energy_hann_schedule":
                control_model.set_control_energy_hann_schedule(
                    control_vector,
                    self.control_config["tau"],
                )
            elif schedule == "middle_layer":
                control_model.set_control_on_middle_layer(
                    control_vector, self.control_config["max_coeff"]
                )
            elif schedule == "control_for_a_given_layer":
                control_model.set_control_for_a_given_layer(
                    control_vector,
                    self.control_config["layer_id"],
                    self.control_config["coeff"],
                )
            elif schedule == "directional_cosine_gated_for_layer":
                control_model.set_control_directional_cosine_gated_for_layer(
                    control_vector,
                    layer_id=self.control_config["layer_id"],
                    coeff=self.control_config["coeff"],
                )
            elif schedule == "directional_relu_gated_for_layer":
                control_model.set_control_directional_relu_gated_for_layer(
                    control_vector,
                    layer_id=self.control_config["layer_id"],
                    coeff=self.control_config["coeff"],
                )
            elif schedule == "directional_sigmoid_gated_for_layer":
                control_model.set_control_directional_sigmoid_gated_for_layer(
                    control_vector,
                    layer_id=self.control_config["layer_id"],
                    coeff=self.control_config["coeff"],
                    beta=self.control_config.get("beta", 1.0),
                )
            elif schedule == "gaussian_random":
                control_model.set_gaussian_random(
                    control_vector,
                    self.control_config["peak_coeff"],
                    std_dev_factor=self.control_config.get("std_dev_factor", 0.25),
                )
            elif schedule == "uniform_gaussian":
                control_model.set_uniform_gaussian(
                    control_vector,
                    self.control_config["peak_coeff"],
                    std_dev_factor=self.control_config.get("std_dev_factor", 0.25),
                )
            elif schedule == "harp_bump_gaussian":
                control_model.set_harp_bump_gaussian(
                    control_vector,
                    self.control_config["peak_coeff"],
                    std_dev_factor=self.control_config.get("std_dev_factor", 0.25),
                )
            else:
                raise ValueError(f"Unknown schedule: {schedule}")

            model = control_model

        self.model = model
        self.tokenizer = tokenizer

    
    # ---------------------------
    # Chat templating / inputs
    # ---------------------------

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Build a single prompt string using the model's chat template.
        Qwen3 'Thinking' variants: enable_thinking=True.
        Non-thinking Qwen3 Instruct variants: explicitly disable thinking to avoid stray tags.
        """
        kwargs = {"add_generation_prompt": True}  # HF best practice for chat models
        if _is_qwen(self.model_name):
            # Decide thinking mode based on model id
            if _is_qwen_thinking_variant(self.model_name):
                kwargs["enable_thinking"] = True  # thinking-only models support this
            else:
                # Instruct-2507 is non-thinking; being explicit is harmless
                kwargs["enable_thinking"] = False  # model card says it's non-thinking

        try:
            chat_template = self.tokenizer.apply_chat_template(
                messages, tokenize=False, **kwargs
            )
            return chat_template
        except:
            return "\n".join(
                [f"[INST] {message['content']} [/INST]" for message in messages]
            )

    def _prepare_inputs_single(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        prompt = self._apply_chat_template(messages)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in encoded.items()}
        input_len = inputs["input_ids"].shape[1]
        return inputs, input_len

    def _prepare_inputs_batch(
        self, batch: List[List[Dict[str, str]]]
    ) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        prompts = [self._apply_chat_template(msgs) for msgs in batch]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in encoded.items()}
        input_lens = [
            int(l)
            for l in (encoded["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)
        ]
        return inputs, input_lens

    # ---------------------------
    # EOS / stopping logic
    # ---------------------------

    def _extra_stop_ids(self) -> List[int]:
        """
        Add model-specific extra stop IDs (e.g., Llama-3 <|eot_id|>).
        """
        ids: List[int] = []
        # Llama-3 turns end with <|eot_id|>; include it if the tokenizer knows it
        if _is_llama(self.model_name):
            try:
                tok_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if isinstance(tok_id, int) and tok_id >= 0:
                    ids.append(tok_id)
            except Exception:
                pass

        # Qwen historic chat tokens; include if present
        if _is_qwen(self.model_name):
            for tok in ("<|im_end|>",):
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(tok)
                    if isinstance(tid, int) and tid >= 0:
                        ids.append(tid)
                except Exception:
                    pass
        return list(dict.fromkeys(ids))  # de-duplicate, preserve order

    def _eos_token_ids(self) -> List[int]:
        base = []
        if self.tokenizer.eos_token_id is not None:
            base.append(int(self.tokenizer.eos_token_id))
        base.extend(self._extra_stop_ids())
        return list(dict.fromkeys(base))

    # ---------------------------
    # Response cleaning.
    # ---------------------------

    def _clean_response(self, text: str):
        if "user\n" in text and "assistant\n" in text:
            text = text.split("assistant\n")[-1]

        # Clean for GPT
        if "assistantfinal" in text:
            text = text.split("assistantfinal")[-1]
        return text.strip()

    # ---------------------------
    # Qwen thinking stripping
    # ---------------------------

    _think_block_re = re.compile(
        r"<think>.*?</think>\s*", flags=re.DOTALL | re.IGNORECASE
    )

    def _strip_qwen_thinking(self, text: str) -> str:
        """
        Remove any <think>...</think> blocks. Also handle the edge case where only </think> appears.
        """
        out = self._think_block_re.sub("", text).strip()
        # If someone outputs trailing `</think>` without opening tag (documented quirk), drop prefix up to it.
        # (We do a conservative cleanup.)
        closing = "</think>"
        i = out.lower().find(closing)
        if i != -1:
            out = out[i + len(closing) :].lstrip()
        return out

    # ---------------------------
    # Public, synchronous API
    # ---------------------------

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        output_format_chat: Optional[bool] = None,
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Generate one completion synchronously.
        """
        max_new = max_tokens if max_tokens is not None else self.max_tokens
        temp = self.temperature if temperature is None else temperature
        want_chat = (
            self.output_format_chat
            if output_format_chat is None
            else output_format_chat
        )

        inputs, in_len = self._prepare_inputs_single(messages)

        gen_kwargs = {
            "max_new_tokens": max_new,
            "temperature": temp,
            "do_sample": (temp is not None and temp > 0),
            "eos_token_id": self._eos_token_ids(),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            gen_kwargs["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty

        outputs = self.model.generate(**inputs, **gen_kwargs)

        # Slice off the prompt for clean decode
        gen_ids = outputs[0, in_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Qwen thinking cleanup if applicable
        if _is_qwen(self.model_name):
            if _is_qwen_thinking_variant(self.model_name):
                text = self._strip_qwen_thinking(text)

        text = self._clean_response(text)

        if want_chat:
            return messages + [{"role": "assistant", "content": text}]
        return text

    def batch_generate(
        self,
        batch: List[List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        output_format_chat: Optional[bool] = None,
    ) -> List[Union[str, List[Dict[str, str]]]]:
        """
        Generate for a batch of chat histories synchronously (order-preserving).
        """
        if not batch:
            return []

        max_new = max_tokens if max_tokens is not None else self.max_tokens
        temp = self.temperature if temperature is None else temperature
        want_chat = (
            self.output_format_chat
            if output_format_chat is None
            else output_format_chat
        )

        inputs, in_lens = self._prepare_inputs_batch(batch)

        gen_kwargs = {
            "max_new_tokens": max_new,
            "temperature": temp,
            "do_sample": (temp is not None and temp > 0),
            "eos_token_id": self._eos_token_ids(),
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            gen_kwargs["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = self.repetition_penalty

        outputs = self.model.generate(**inputs, **gen_kwargs)

        results: List[Union[str, List[Dict[str, str]]]] = []
        for i in range(outputs.size(0)):
            out_ids = outputs[i, in_lens[i] :]
            text = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()

            if _is_qwen(self.model_name):
                if _is_qwen_thinking_variant(self.model_name):
                    text = self._strip_qwen_thinking(text)

            text = self._clean_response(text)

            if want_chat:
                results.append(batch[i] + [{"role": "assistant", "content": text}])
            else:
                results.append(text)

        return results
