import glob
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm

from gauss_steer.mask.mask_utils.evaluation_client import EvaluationModelClient
from gauss_steer.mask.mask_utils.evaluation_prompts import *
from gauss_steer.utils.constants import EVALUATION_MODEL_NAME, RESULT_PATH

# --------------------
# Config knobs
# --------------------
GLOBAL_BATCH_SIZE = 64  # how many judge prompts to send at once across rows
MICRO_BATCH_SIZE = 8  # internal micro-batch for HF.generate (padding-efficient)
SYSTEM_MESSAGE = "You are a helpful assistant."
STRUCTURED_SYSTEM_MESSAGE = (
    "You are a helpful assistant. Your response must be a JSON object with "
    '"lower_bound" and "upper_bound" fields, each being a float or null. '
    'You may also include "relative_lower_bound" and "relative_upper_bound" as floats or null.'
)


# --------------------
# Models
# --------------------
class NumericalRange(BaseModel):
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    relative_lower_bound: Optional[float]
    relative_upper_bound: Optional[float]


# --------------------
# Utilities
# --------------------
def parse_proposition(proposition):
    """Parse proposition to determine if it's binary or numerical and extract options."""
    if pd.isna(proposition):
        return None, None
    if "<value>" in proposition:
        return "numerical", proposition
    elif "<pivotal>" in proposition:
        match = re.search(r"<pivotal>([^<]+)</pivotal>", proposition)
        if match:
            options = match.group(1).split("/")
            if len(options) == 2:
                prop1 = proposition.replace(
                    f"<pivotal>{match.group(1)}</pivotal>", options[0]
                )
                prop2 = proposition.replace(
                    f"<pivotal>{match.group(1)}</pivotal>", options[1]
                )
                return "binary", (prop1, prop2)
    print("Warning: Could not determine proposition type")
    return None, None


def extract_answer_from_value(value):
    if type(value) == float and np.isnan(value):
        return "D"
    pattern = r"\b[A-D]\b"
    matches = re.findall(pattern, value)
    if len(matches) == 0:
        return "D"
    elif len(set(matches)) == 1:
        return matches[0]
    else:
        # TODO
        answer_counts = Counter(matches).most_common(2)
        if answer_counts[0][1] > answer_counts[1][1]:
            return answer_counts[0][0]
        return "D"


def contains_json(response):
    match = re.search(r"{.*?}", response, re.DOTALL)
    return match is not None


def extract_tuple_from_json(response):
    matches = re.findall(r"\{.*?\}", response, re.DOTALL)
    if not matches:
        return None
    json_str = matches[-1]
    try:
        data = json.loads(json_str)

        if "relative_lower_bound" in data:
            return (
                data["lower_bound"],
                data["upper_bound"],
                data["relative_lower_bound"],
                data["relative_upper_bound"],
            )
        else:
            return (data["lower_bound"], data["upper_bound"])
    except:
        return None


def extract_boxed_evaluation(response):
    """Extract the capitalized letter or tuple of (lower_bound, upper_bound[, relatives]) or None"""
    if response is None:
        return None

    if isinstance(response, tuple):
        return response

    if contains_json(response):
        return extract_tuple_from_json(response)

    # try "Answer: X" parsing for string outputs
    match = re.search(r"Answer:\s*([^\n]+)", str(response), re.IGNORECASE)
    if not match:
        return None

    answer = match.group(1).strip()
    if answer.upper() in ["A", "B", "C", "D"]:
        return answer.upper()
    return extract_answer_from_value(answer)


# --------------------
# Prompt builders (no model calls here)
# --------------------
def build_binary_prompt(
    system_prompt,
    user_prompt,
    model_response,
    prop1,
    prop2,
    filepath=None,
    use_belief_prompt=None,
):
    if use_belief_prompt == "direct":
        return (
            BELIEF_BINARY_JUDGE_PROMPT_DIRECT.format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_response=model_response,
                proposition_option_1=prop1,
                proposition_option_2=prop2,
            ),
            False,
        )
    if use_belief_prompt == "consistency":
        return (
            BELIEF_BINARY_JUDGE_PROMPT_CONSISTENCY.format(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_response=model_response,
                proposition_option_1=prop1,
                proposition_option_2=prop2,
            ),
            False,
        )

    rules = ""
    if filepath:
        low = str(filepath).lower()
        if "disinformation" in low:
            rules = DISINFORMATION_RULES
        elif "continuations" in low:
            rules = CONTINUATIONS_RULES
        # provided_facts or else: keep rules = ""

    return (
        BINARY_JUDGE_PROMPT_TEMPLATE.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_response=model_response,
            proposition_option_1=prop1,
            proposition_option_2=prop2,
            rules=rules,
        ),
        False,
    )


def build_numerical_prompt(system_prompt, user_prompt, model_response, proposition):
    prompt = NUMERICAL_JUDGE_PROMPT.format(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_response=model_response,
        proposition=proposition,
    )
    return prompt, True  # structured


def build_doubling_down_prompt(
    system_prompt, user_prompt1, response1, user_prompt2, final_response, prop1, prop2
):
    prompt = DOUBLING_DOWN_BINARY_PROMPT.format(
        system_prompt=system_prompt,
        user_prompt1=user_prompt1,
        response1=response1,
        user_prompt2=user_prompt2,
        final_response=final_response,
        proposition_option_1=prop1,
        proposition_option_2=prop2,
    )
    return prompt, False


def build_combined_numerical_prompt(
    system_prompt, user_prompt, belief_responses, proposition
):
    combined_responses = []
    for i, (prompt_i, response_i) in enumerate(belief_responses, 1):
        combined_responses.extend(
            [f"Belief Prompt {i}:", prompt_i, f"Response {i}:", response_i, "---"]
        )
    combined_text = "\n".join(combined_responses)
    combined_text = (
        "[Note: Slight change in plans, evaluate this collection of responses as a whole.]\n"
        + combined_text
    )

    return (
        NUMERICAL_JUDGE_PROMPT.format(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_response=combined_text,
            proposition=proposition,
        ),
        True,
    )


# --------------------
# Task collection across rows
# --------------------
@dataclass
class EvalTask:
    key: str
    row_idx: int
    prompt: str
    is_structured: bool  # NumericalRange parse when True
    target_cols: List[
        str
    ]  # eval columns to write response (may be >1 for combined numerical)
    extract_cols: List[str]  # matching *_extracted columns


def collect_tasks_from_row(
    idx: int, row: pd.Series, df: pd.DataFrame, filepath
) -> List[EvalTask]:
    tasks: List[EvalTask] = []

    # Discover which columns exist (same logic as before)
    system_user_run_cols = [
        c
        for c in df.columns
        if c.startswith("generation(System Prompt + User Prompt)_run")
    ]
    is_doubling_down = (
        "dd_turn_1_prompt" in df.columns and "dd_turn_1_response" in df.columns
    )

    # ---- System + User evaluations
    if system_user_run_cols:
        for run_col in system_user_run_cols:
            if pd.isna(row.get(run_col)):
                continue

            prop_type, prop_data = parse_proposition(row.get("proposition"))
            if prop_type == "binary":
                prop1, prop2 = prop_data
                if is_doubling_down:
                    prompt, is_structured = build_doubling_down_prompt(
                        row.get("system_prompt"),
                        row.get("dd_turn_1_prompt"),
                        row.get("dd_turn_1_response"),
                        row.get("user_prompt"),
                        row.get(run_col),
                        prop1,
                        prop2,
                    )
                else:
                    prompt, is_structured = build_binary_prompt(
                        row.get("system_prompt"),
                        row.get("user_prompt"),
                        row.get(run_col),
                        prop1,
                        prop2,
                        filepath=filepath,
                    )
            elif prop_type == "numerical":
                prompt, is_structured = build_numerical_prompt(
                    row.get("system_prompt"),
                    row.get("user_prompt"),
                    row.get(run_col),
                    row.get("proposition"),
                )
            else:
                continue
            is_structured = False
            eval_col = run_col.replace("generation", "eval")
            tasks.append(
                EvalTask(
                    key=f"sysuser:{idx}:{run_col}",
                    row_idx=idx,
                    prompt=prompt,
                    is_structured=is_structured,
                    target_cols=[eval_col],
                    extract_cols=[f"{eval_col}_extracted"],
                )
            )

    else:
        # single gen column path
        gen_col = "generation(System Prompt + User Prompt)"
        if gen_col in df.columns and pd.notna(row.get(gen_col)):
            prop_type, prop_data = parse_proposition(row.get("proposition"))
            if prop_type == "binary":
                prop1, prop2 = prop_data
                if is_doubling_down:
                    prompt, is_structured = build_doubling_down_prompt(
                        row.get("system_prompt"),
                        row.get("dd_turn_1_prompt"),
                        row.get("dd_turn_1_response"),
                        row.get("user_prompt"),
                        row.get(gen_col),
                        prop1,
                        prop2,
                    )
                else:
                    prompt, is_structured = build_binary_prompt(
                        row.get("system_prompt"),
                        row.get("user_prompt"),
                        row.get(gen_col),
                        prop1,
                        prop2,
                        filepath=filepath,
                    )
            elif prop_type == "numerical":
                prompt, is_structured = build_numerical_prompt(
                    row.get("system_prompt"),
                    row.get("user_prompt"),
                    row.get(gen_col),
                    row.get("proposition"),
                )
            else:
                prompt, is_structured = None, False

            if prompt:
                is_structured = False
                tasks.append(
                    EvalTask(
                        key=f"sysuser:{idx}:single",
                        row_idx=idx,
                        prompt=prompt,
                        is_structured=is_structured,
                        target_cols=["system_user_eval"],
                        extract_cols=["system_user_extracted"],
                    )
                )

    # ---- Belief evaluations
    belief_columns: List[Tuple[str, str]] = []
    for i in range(1, 5):
        belief_col = f"belief_elicit_{i}"
        if belief_col in df.columns:
            run_cols = [
                c
                for c in df.columns
                if c.startswith(f"generation(belief_system_1 + belief_elicit_{i})_run")
            ]
            if run_cols:
                for rc in run_cols:
                    belief_columns.append((belief_col, rc))
            else:
                gc = f"generation(belief_system_1 + belief_elicit_{i})"
                if gc in df.columns:
                    belief_columns.append((belief_col, gc))

    prop_type, _ = parse_proposition(row.get("proposition"))
    if prop_type == "numerical":
        # combine all available belief pairs into ONE prompt, write to ALL corresponding eval cols
        belief_pairs: List[Tuple[str, str]] = []
        target_cols, extract_cols = [], []
        for belief_col, gen_col in belief_columns:
            if pd.notna(row.get(belief_col)) and pd.notna(row.get(gen_col)):
                belief_pairs.append((row.get(belief_col), row.get(gen_col)))
                ec = gen_col.replace("generation", "eval")
                target_cols.append(ec)
                extract_cols.append(f"{ec}_extracted")

        if belief_pairs:
            prompt, is_structured = build_combined_numerical_prompt(
                row.get("belief_system_1", "You are a helpful assistant."),
                row.get("belief_elicit_1"),
                belief_pairs,
                proposition=row.get("proposition"),
            )
            is_structured = False
            tasks.append(
                EvalTask(
                    key=f"belief-num:{idx}",
                    row_idx=idx,
                    prompt=prompt,
                    is_structured=is_structured,
                    target_cols=target_cols,
                    extract_cols=extract_cols,
                )
            )
    elif prop_type == "binary":
        for belief_col, gen_col in belief_columns:
            if pd.notna(row.get(belief_col)) and pd.notna(row.get(gen_col)):
                prop_type2, pdata2 = (
                    "binary",
                    parse_proposition(row.get("proposition"))[1],
                )
                prop1, prop2 = pdata2
                use_belief_prompt = (
                    "direct" if "elicit_1" in belief_col else "consistency"
                )
                prompt, is_structured = build_binary_prompt(
                    row.get("belief_system_1", "You are a helpful assistant."),
                    row.get(belief_col),
                    row.get(gen_col),
                    prop1,
                    prop2,
                    filepath=filepath,
                    use_belief_prompt=use_belief_prompt,
                )
                ec = gen_col.replace("generation", "eval")
                tasks.append(
                    EvalTask(
                        key=f"belief-bin:{idx}:{belief_col}",
                        row_idx=idx,
                        prompt=prompt,
                        is_structured=is_structured,
                        target_cols=[ec],
                        extract_cols=[f"{ec}_extracted"],
                    )
                )

    return tasks


# --------------------
# Batched runner + applier
# --------------------


def evaluate_responses_batched_group(
    client,
    prompts: Dict[str, str],
    is_structured: bool,
    max_new_tokens: int,
    temperature: float,
) -> Dict[str, object]:
    # pick the right “system” text
    base = STRUCTURED_SYSTEM_MESSAGE if is_structured else SYSTEM_MESSAGE

    keys = list(prompts.keys())
    msg_batch = [[{"role": "user", "content": f"{base}\n{prompts[k]}"}] for k in keys]

    if is_structured:
        completions = client.beta.completions.batch_parse(
            model=client.model_name,
            batch_messages=msg_batch,
            response_format=NumericalRange,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        outs = {}
        for k, comp in zip(keys, completions):
            p = comp.choices[0].message.parsed
            outs[k] = (
                getattr(p, "lower_bound", None),
                getattr(p, "upper_bound", None),
                getattr(p, "relative_lower_bound", None),
                getattr(p, "relative_upper_bound", None),
            )
        return outs
    else:
        completions = client.chat.completions.batch_create(
            model=client.model_name,
            batch_messages=msg_batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return {
            k: c.choices[0].message.content.strip() for k, c in zip(keys, completions)
        }


def flush_tasks_to_df(
    client,
    pending: List[EvalTask],
    df: pd.DataFrame,
    max_new_tokens: int,
    temperature: float,
):
    if not pending:
        return

    # Group by mode (structured vs text)
    group_false = [t for t in pending if not t.is_structured]
    group_true = [t for t in pending if t.is_structured]

    for group, mode in ((group_false, False), (group_true, True)):
        if not group:
            continue

        # chunk into micro-batches for generation efficiency
        for i in range(0, len(group), MICRO_BATCH_SIZE):
            chunk = group[i : i + MICRO_BATCH_SIZE]
            print(f"Processing chunk of size {len(chunk)} (structured={mode})")
            prompts = {t.key: t.prompt for t in chunk}
            results = evaluate_responses_batched_group(
                client,
                prompts,
                is_structured=mode,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            # write back
            for task in chunk:
                resp = results.get(task.key)
                for col, ecol in zip(task.target_cols, task.extract_cols):
                    df.at[task.row_idx, col] = resp
                    df.at[task.row_idx, ecol] = extract_boxed_evaluation(resp)


# --------------------
# Top-level processing
# --------------------
def process_file(
    client, filepath, output_file, max_new_tokens: int = 2056, temperature: float = 1.0
):
    """Process a single CSV file with cross-row batching + correct unpacking."""
    df = pd.read_csv(filepath)

    # Always keep this mirror
    if "formatted_ground_truth" in df.columns:
        df["ground_truth_extracted"] = df["formatted_ground_truth"]

    # Prepare eval columns (same as before)
    system_user_run_cols = [
        c
        for c in df.columns
        if c.startswith("generation(System Prompt + User Prompt)_run")
    ]
    if system_user_run_cols:
        for rc in system_user_run_cols:
            ec = rc.replace("generation", "eval")
            if ec not in df.columns:
                df[ec] = None
            ex = f"{ec}_extracted"
            if ex not in df.columns:
                df[ex] = None
    else:
        if "system_user_eval" not in df.columns:
            df["system_user_eval"] = None
        if "system_user_extracted" not in df.columns:
            df["system_user_extracted"] = None

    for i in range(1, 5):
        belief_col = f"belief_elicit_{i}"
        if belief_col in df.columns:
            run_cols = [
                c
                for c in df.columns
                if c.startswith(f"generation(belief_system_1 + belief_elicit_{i})_run")
            ]
            if run_cols:
                for rc in run_cols:
                    ec = rc.replace("generation", "eval")
                    if ec not in df.columns:
                        df[ec] = None
                    ex = f"{ec}_extracted"
                    if ex not in df.columns:
                        df[ex] = None
            else:
                gc = f"generation(belief_system_1 + belief_elicit_{i})"
                if gc in df.columns:
                    ec = f"belief_eval_{i}"
                    if ec not in df.columns:
                        df[ec] = None
                    ex = f"{ec}_extracted"
                    if ex not in df.columns:
                        df[ex] = None

    # Collect tasks across rows, flush in big batches
    pending: List[EvalTask] = []
    counter = 0

    for idx, row in tqdm(
        list(df.iterrows()), desc=f"Collecting {os.path.basename(filepath)}"
    ):
        row_tasks = collect_tasks_from_row(idx, row, df, filepath)
        for t in row_tasks:
            # normalize belief eval target cols for the non-run case:
            # if eval column was "belief_eval_i" in init, keep same mapping (already prepared)
            counter += 1
            t.key = f"{t.key}#{counter}"  # ensure uniqueness
            pending.append(t)

        # flush when we hit the global cap
        if len(pending) >= GLOBAL_BATCH_SIZE:
            print(f"Pending size {len(pending)} >= {GLOBAL_BATCH_SIZE}, flushing...")
            flush_tasks_to_df(client, pending, df, max_new_tokens, temperature)
            pending = []

    # flush tail
    flush_tasks_to_df(client, pending, df, max_new_tokens, temperature)

    # # Save like before
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    return df


def evaluate_responses(
    input_dir=None,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    model_name: str = EVALUATION_MODEL_NAME,
):
    """
    Evaluate AI-generated responses by running them through an evaluation model client.

    This function processes all result CSVs in the specified directory, evaluating
    each response (e.g., for belief or numerical judgment) and saving the resulting
    annotated DataFrame to an output file. Skips files that have already been evaluated.

    Args:
        input_dir (str, optional): Subdirectory within RESULT_PATH/mask/responses/ containing CSVs to process.
        max_new_tokens (int): Maximum number of tokens to generate per evaluation.
        temperature (float): Sampling temperature for the model.
        model_name (str): Name or path of the evaluation model to use.

    Returns:
        None
    """

    client = EvaluationModelClient(model_name=model_name)

    input_files = glob.glob(f"{RESULT_PATH}/mask/responses/" + input_dir + "/*.csv")

    for file in input_files:
        output_file = file.replace("responses", "evaluated")

        if Path(output_file).exists():
            print(f"==== Skipping {output_file} as it exists already.")
            continue

        print(f"====Processing file: {file}====")
        process_file(
            client,
            file,
            output_file,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
