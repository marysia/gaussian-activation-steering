from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

from gauss_steer.utils.constants import DATA_PATH, RESULT_PATH
from gauss_steer.utils.model import HuggingFaceModelClient


def process_row(
    row: pd.Series,
    index: int,
    is_doubling_down: bool,
    lie_k: int,
    belief_k: Dict[str, int],
    df: pd.DataFrame,
) -> Tuple[List[List[dict]], List[int], List[str]]:
    """
    Build all chat-histories for this row.
    Returns (row_prompts, row_indices, row_prompt_types)
    """
    row_prompts: List[List[dict]] = []
    row_indices: List[int] = []
    row_prompt_types: List[str] = []

    # Lying / main prompts
    if is_doubling_down:
        for run in range(lie_k):
            row_prompts.append(
                [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["dd_turn_1_prompt"]},
                    {"role": "assistant", "content": row["dd_turn_1_response"]},
                    {"role": "user", "content": row["user_prompt"]},
                ]
            )
            row_indices.append(index)
            row_prompt_types.append(f"lying_run{run+1}")
    else:
        for run in range(lie_k):
            row_prompts.append(
                [
                    {"role": "system", "content": row["system_prompt"]},
                    {"role": "user", "content": row["user_prompt"]},
                ]
            )
            row_indices.append(index)
            row_prompt_types.append(f"lying_run{run+1}")

    # Belief elicitation (optional)
    if "belief_elicit_1" in df.columns:

        for be_num, k_value in belief_k.items():
            belief_col = f"belief_elicit_{be_num}"
            if pd.notna(row.get(belief_col)):
                for run in range(k_value):
                    row_prompts.append(
                        [
                            {"role": "system", "content": row["belief_system_1"]},
                            {"role": "user", "content": row[belief_col]},
                        ]
                    )
                    row_indices.append(index)
                    row_prompt_types.append(f"be{be_num}_run{run+1}")

    return row_prompts, row_indices, row_prompt_types


def process_dataframe_sync(
    client: HuggingFaceModelClient,
    input_csv: str,
    output_csv: str,
    belief_k: Dict[str, int],
    lie_k: int = 1,
    batch_size: int = 64,
    validation: bool = True,
    max_new_tokens: int = 1000,
    temperature: float = 1.0,
):
    """
    Synchronous, order-preserving batched processing with HuggingFaceModelClient.
    - Chat templating + EOS are handled in the client to support Llama, Mistral/Mixtral, Qwen.
    """
    df = pd.read_csv(input_csv)

    if validation:
        n = max(1, int(0.2 * len(df)))
        df = df.sample(n, random_state=0)

    df = df.head(2)
    print(f"{input_csv} has {len(df)} rows (validation={validation})")

    is_doubling_down = (
        "dd_turn_1_prompt" in df.columns and "dd_turn_1_response" in df.columns
    )
    print(f"Is doubling down? {is_doubling_down}")

    # Collect all prompts
    all_prompts: List[List[dict]] = []
    indices: List[int] = []
    prompt_types: List[str] = []

    if "belief_system_1" not in df.columns:
        df["belief_system_1"] = "You are a helpful assistant."

    for index, row in df.iterrows():
        row_prompts, row_indices, row_prompt_types = process_row(
            row, index, is_doubling_down, lie_k, belief_k, df
        )
        all_prompts.extend(row_prompts)
        indices.extend(row_indices)
        prompt_types.extend(row_prompt_types)

    # Output column names
    column_mapping = {
        f"lying_run{run+1}": f"generation(System Prompt + User Prompt)_run{run+1}"
        for run in range(lie_k)
    }
    if "belief_elicit_1" in df.columns:
        for be_num, k_value in belief_k.items():
            for run in range(k_value):
                column_mapping[f"be{be_num}_run{run+1}"] = (
                    f"generation(belief_system_1 + belief_elicit_{be_num})_run{run+1}"
                )

    # Ensure columns exist
    for col in column_mapping.values():
        if col not in df.columns:
            df[col] = pd.NA

    # Batched generation (order preserved)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    for batch_start in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[batch_start : batch_start + batch_size]
        batch_indices = indices[batch_start : batch_start + batch_size]
        batch_types = prompt_types[batch_start : batch_start + batch_size]

        print(
            f"Generating batch {batch_start//batch_size + 1} "
            f"({len(batch_prompts)} prompts)"
        )

        try:
            batch_texts = client.batch_generate(
                batch_prompts,
                max_tokens=max_new_tokens,
                temperature=temperature,
                output_format_chat=False,
            )
            # batch_generate returns a list of strings (since output_format_chat=False)
            assert len(batch_texts) == len(batch_prompts)
        except Exception as e:
            batch_texts = [f"[ERROR: {e}]" for _ in batch_prompts]

        # Write back to the correct rows/columns
        for resp_text, idx, ptype in zip(batch_texts, batch_indices, batch_types):
            out_col = column_mapping.get(ptype)
            if out_col is not None:
                df.at[idx, out_col] = resp_text

        # Incremental checkpoint
        df.to_csv(output_csv, index=False)

    # Final save
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")


def generate_responses(
    client: HuggingFaceModelClient,
    output_folder: str,
    identifier: str,
    batch_size: int = 128,
    lie_k: int = 1,
    validation: bool = True,
    max_new_tokens: int = 1000,
    temperature: float = 1.0,
):
    """
    Walk DATA_PATH/mask/*.csv, generate, and save to RESULT_PATH/mask/responses/<output_folder>/.
    """
    data_dir = DATA_PATH / "mask/"
    input_files = list(data_dir.glob("*.csv"))

    for file in tqdm(input_files, desc="Processing CSV files"):
        # Example heuristic — keep your original behavior
        belief_k = {"1": 3} if "statistics" in str(file) else {"1": 3, "2": 1, "3": 1}

        file_name = file.name.replace(".csv", f"_{identifier}.csv")
        output_file = f"{RESULT_PATH}/mask/responses/{output_folder}/{file_name}"
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.exists():
            df = pd.read_csv(output_path)
            print(f"Skipping {output_path} (exists; length={len(df)})")
            continue

        print("Processing", file, "→", output_path)
        process_dataframe_sync(
            client=client,
            input_csv=str(file),
            output_csv=str(output_path),
            belief_k=belief_k,
            lie_k=lie_k,
            batch_size=batch_size,
            validation=validation,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    print("Done (batched, sync).")
