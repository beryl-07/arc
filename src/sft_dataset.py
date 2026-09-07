"""Build stratified conflict splits for SFT and GRPO."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


LOG = logging.getLogger(__name__)
DEFAULT_TARGET_SFT_SIZE = 750
DEFAULT_MIN_PER_CATEGORY = 20
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Loading and normalization
# ---------------------------------------------------------------------------


def read_scenarios(path: Path) -> pd.DataFrame:
    """Load a JSONL scenarios file and add the normalized conflict category."""

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_json(path, lines=True)
    if "query_id" not in df.columns:
        raise ValueError("Missing required column: query_id")
    if "conflicts" not in df.columns:
        raise ValueError("Missing required column: conflicts")

    if df["query_id"].duplicated().any():
        duplicate_ids = df.loc[df["query_id"].duplicated(), "query_id"].tolist()
        raise ValueError(f"Duplicate query_id values found: {duplicate_ids[:10]}")

    df = df.copy()
    df["_source_index"] = np.arange(len(df))
    df["conflict_category"] = df["conflicts"].apply(normalize_conflict_category)
    return df


def normalize_conflict_category(conflicts: Any) -> str:
    """Convert the conflict list into a stable stratification label."""

    if not conflicts:
        return "sans_conflit"
    if not isinstance(conflicts, list):
        raise ValueError("Each 'conflicts' value must be a list.")

    conflict_types: list[str] = []
    for conflict in conflicts:
        if not isinstance(conflict, dict):
            raise ValueError("Each conflict entry must be an object.")
        conflict_type = conflict.get("type")
        if conflict_type is None:
            raise ValueError("Each conflict entry must contain a 'type' field.")
        conflict_types.append(str(conflict_type))

    normalized = sorted(set(conflict_types))
    return "+".join(normalized) if normalized else "sans_conflit"


# ---------------------------------------------------------------------------
def split_global(
    df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the 80/10/10 global split stratified by conflict_category."""

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    indices = np.arange(len(df))
    train_val_idx, test_idx = next(splitter.split(indices, df["conflict_category"]))
    train_val_df = df.iloc[train_val_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_val_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.10 / 0.90,
        random_state=seed,
    )
    train_idx, val_idx = next(train_val_splitter.split(np.arange(len(train_val_df)), train_val_df["conflict_category"]))
    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# SFT / GRPO split
# ---------------------------------------------------------------------------


def allocate_proportionally(
    capacities: dict[str, int],
    total_slots: int,
) -> dict[str, int]:
    """Allocate integer slots proportionally with largest-remainder rounding."""

    if total_slots <= 0 or not capacities:
        return {key: 0 for key in capacities}

    total_capacity = sum(capacities.values())
    if total_slots >= total_capacity:
        return dict(capacities)

    raw = {key: total_slots * capacity / total_capacity for key, capacity in capacities.items()}
    allocation = {key: min(capacities[key], int(np.floor(value))) for key, value in raw.items()}
    remaining = total_slots - sum(allocation.values())

    if remaining > 0:
        ranking = sorted(
            capacities.keys(),
            key=lambda key: (
                -(raw[key] - allocation[key]),
                -capacities[key],
                key,
            ),
        )
        for key in ranking:
            if remaining == 0:
                break
            if allocation[key] < capacities[key]:
                allocation[key] += 1
                remaining -= 1
    return allocation


def split_train_for_sft_and_grpo(
    train_df: pd.DataFrame,
    target_sft_size: int = DEFAULT_TARGET_SFT_SIZE,
    min_per_category: int = DEFAULT_MIN_PER_CATEGORY,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the train set into SFT and GRPO with the requested oversampling rule."""

    if "conflict_category" not in train_df.columns:
        raise ValueError("train_df must contain conflict_category")

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = []

    sans_df = train_df[train_df["conflict_category"] == "sans_conflit"]
    selected_indices.extend(sans_df.index.tolist())

    remaining_budget = target_sft_size - len(sans_df)
    non_sans = train_df[train_df["conflict_category"] != "sans_conflit"]
    category_counts = non_sans["conflict_category"].value_counts().sort_index()

    if remaining_budget <= 0 or category_counts.empty:
        sft_df = train_df.loc[sorted(set(selected_indices))].copy()
        grpo_df = train_df.drop(index=sft_df.index).copy()
        return sort_by_source_index(sft_df), sort_by_source_index(grpo_df)

    fixed_allocations = {cat: min(count, min_per_category) for cat, count in category_counts.items()}
    fixed_total = sum(fixed_allocations.values())

    for category, take_n in fixed_allocations.items():
        if take_n <= 0:
            continue
        pool = train_df.index[train_df["conflict_category"] == category].to_numpy()
        chosen = rng.choice(pool, size=take_n, replace=False)
        selected_indices.extend(chosen.tolist())

    remaining_budget = target_sft_size - len(selected_indices)
    if remaining_budget <= 0:
        sft_df = train_df.loc[sorted(set(selected_indices))].copy()
        grpo_df = train_df.drop(index=sft_df.index).copy()
        return sort_by_source_index(sft_df), sort_by_source_index(grpo_df)

    remaining_capacities = {
        category: count - fixed_allocations[category]
        for category, count in category_counts.items()
        if count > fixed_allocations[category]
    }

    if not remaining_capacities:
        sft_df = train_df.loc[sorted(set(selected_indices))].copy()
        grpo_df = train_df.drop(index=sft_df.index).copy()
        return sort_by_source_index(sft_df), sort_by_source_index(grpo_df)

    extra_allocations = allocate_proportionally(remaining_capacities, remaining_budget)
    for category, take_n in extra_allocations.items():
        if take_n <= 0:
            continue
        already_selected = set(selected_indices)
        pool = train_df.index[
            (train_df["conflict_category"] == category) & (~train_df.index.isin(already_selected))
        ].to_numpy()
        take_n = min(take_n, len(pool))
        if take_n <= 0:
            continue
        chosen = rng.choice(pool, size=take_n, replace=False)
        selected_indices.extend(chosen.tolist())

    sft_df = train_df.loc[sorted(set(selected_indices))].copy()
    grpo_df = train_df.drop(index=sft_df.index).copy()
    return sort_by_source_index(sft_df), sort_by_source_index(grpo_df)


# ---------------------------------------------------------------------------
# Reporting and verification
# ---------------------------------------------------------------------------


def sort_by_source_index(df: pd.DataFrame) -> pd.DataFrame:
    if "_source_index" not in df.columns:
        return df.reset_index(drop=True)
    return df.sort_values("_source_index").drop(columns=["_source_index"]).reset_index(drop=True)


def distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["conflict_category"].value_counts().sort_index()
    total = len(df)
    table = pd.DataFrame({
        "count": counts,
        "pct": (counts / total * 100).round(2),
    })
    table.index.name = "conflict_category"
    return table


def print_distribution(name: str, df: pd.DataFrame) -> None:
    table = distribution_table(df)
    print(f"\n{name} ({len(df)} rows)")
    print(table.to_string())


def max_distribution_diff_pp(reference: pd.DataFrame, candidate: pd.DataFrame) -> float:
    ref = reference["conflict_category"].value_counts(normalize=True)
    cand = candidate["conflict_category"].value_counts(normalize=True)
    aligned = pd.concat([ref, cand], axis=1, keys=["ref", "cand"]).fillna(0.0)
    return float((aligned["ref"] - aligned["cand"]).abs().max() * 100.0)


def verify_disjointness(splits: dict[str, pd.DataFrame]) -> None:
    seen: dict[str, str] = {}
    for split_name, df in splits.items():
        for query_id in df["query_id"].astype(str):
            if query_id in seen:
                raise ValueError(f"query_id overlap detected: {query_id} in {seen[query_id]} and {split_name}")
            seen[query_id] = split_name
    print("query_id disjointness: OK")


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = df.drop(columns=["_source_index"], errors="ignore").to_dict(orient="records")
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_splits(
    input_path: Path,
    output_dir: Path,
    target_sft_size: int = DEFAULT_TARGET_SFT_SIZE,
    min_per_category: int = DEFAULT_MIN_PER_CATEGORY,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    df = read_scenarios(input_path)
    overall_df = sort_by_source_index(df)

    train_df, val_df, test_df = split_global(df, seed=seed)
    val_df = sort_by_source_index(val_df)
    test_df = sort_by_source_index(test_df)

    sft_df, grpo_df = split_train_for_sft_and_grpo(
        train_df,
        target_sft_size=target_sft_size,
        min_per_category=min_per_category,
        seed=seed,
    )

    splits = {
        "sft_train": sft_df,
        "grpo_train": grpo_df,
        "val": val_df,
        "test": test_df,
    }

    verify_disjointness(splits)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        name: output_dir / f"{name}.jsonl"
        for name in splits
    }
    for name, split_df in splits.items():
        write_jsonl(split_df, output_paths[name])

    print("\nOverall distribution (%)")
    print_distribution("overall", overall_df)
    for name, split_df in splits.items():
        print_distribution(name, split_df)

    print("\nDistribution checks")
    print(
        f"val max abs diff vs overall: {max_distribution_diff_pp(overall_df, val_df):.2f} pp"
    )
    print(
        f"test max abs diff vs overall: {max_distribution_diff_pp(overall_df, test_df):.2f} pp"
    )

    train_sans_total = int((train_df["conflict_category"] == "sans_conflit").sum())
    sans_in_sft = int((sft_df["conflict_category"] == "sans_conflit").sum())
    print(
        f"sft_train sans_conflit coverage within train: {sans_in_sft}/{train_sans_total} "
        f"({(sans_in_sft / train_sans_total * 100 if train_sans_total else 0.0):.2f}%)"
    )
    if sans_in_sft != train_sans_total:
        raise ValueError("sft_train does not contain 100% of sans_conflit samples from the train split.")

    sft_total = len(sft_df)
    grpo_total = len(grpo_df)
    print(f"\nSFT target size: {target_sft_size}")
    print(f"Actual sft_train size: {sft_total}")
    print(f"Actual grpo_train size: {grpo_total}")

    return output_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stratified SFT/GRPO/validation/test splits from scenarios.jsonl."
    )
    parser.add_argument(
        "input_path",
        type=Path,
        nargs="?",
        default=Path("data/scenarios.jsonl"),
        help="Input JSONL file. Defaults to data/scenarios.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory where sft_train.jsonl, grpo_train.jsonl, val.jsonl and test.jsonl are written.",
    )
    parser.add_argument("--target-sft-size", type=int, default=DEFAULT_TARGET_SFT_SIZE)
    parser.add_argument("--min-per-category", type=int, default=DEFAULT_MIN_PER_CATEGORY)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    build_splits(
        input_path=args.input_path,
        output_dir=args.output_dir,
        target_sft_size=args.target_sft_size,
        min_per_category=args.min_per_category,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
