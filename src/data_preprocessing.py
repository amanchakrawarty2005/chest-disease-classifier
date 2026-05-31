"""Phase 1B preprocessing pipeline for NIH Chest X-ray metadata.

This module prepares train/validation/test metadata files used by training.
Outputs are intentionally stable so downstream scripts keep working.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from config import (
    DISEASE_CLASSES,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)

LOGGER = logging.getLogger("data_preprocessing")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_finding_labels(raw_label: str) -> List[str]:
    """Convert NIH 'Finding Labels' cell into a clean disease list."""
    if pd.isna(raw_label) or str(raw_label).strip() == "No Finding":
        return []
    return [item.strip() for item in str(raw_label).split("|") if item.strip()]


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Invalid split ratios: train+val+test must be 1.0, got {total:.6f}")


def safe_stratify_target(series: pd.Series, stage_name: str, min_count: int = 2) -> Optional[pd.Series]:
    """Return series for stratification when every class has enough samples."""
    counts = series.value_counts(dropna=False)
    current_min = int(counts.min()) if len(counts) else 0

    if current_min < min_count:
        LOGGER.warning(
            "%s stratification disabled: smallest group has %d sample(s), need >= %d.",
            stage_name,
            current_min,
            min_count,
        )
        return None

    return series


def build_image_lookup(raw_data_dir: Path, patterns: Iterable[str] = ("*.png", "*.jpg", "*.jpeg")) -> Dict[str, str]:
    """Map image filename -> absolute path (supports nested NIH folders)."""
    lookup: Dict[str, str] = {}
    for pattern in patterns:
        for path in raw_data_dir.rglob(pattern):
            lookup[path.name] = str(path)
    return lookup


def compute_class_weights(train_labels: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    """Compute inverse-frequency weights for multi-label BCE training."""
    if train_labels.ndim != 2:
        raise ValueError("Expected 2D label matrix for class-weight computation.")

    total_samples, num_classes = train_labels.shape
    weights: Dict[str, float] = {}

    for idx, class_name in enumerate(class_names):
        positive_count = int(np.sum(train_labels[:, idx]))
        weights[class_name] = float(total_samples / (num_classes * max(positive_count, 1)))

    return weights


class NIHPreprocessor:
    """Prepare metadata splits and labels for model training/evaluation."""

    def __init__(
        self,
        raw_data_dir: Path,
        processed_data_dir: Path,
        csv_name: str = "Data_Entry_2017.csv",
        seed: int = RANDOM_SEED,
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.csv_path = self.raw_data_dir / csv_name
        self.seed = seed

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.df: Optional[pd.DataFrame] = None

    def load_metadata(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        required_columns = {"Image Index", "Finding Labels"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        self.df = df
        LOGGER.info("Loaded metadata with %s rows from %s", f"{len(df):,}", self.csv_path)
        return df

    def build_label_matrix(self) -> np.ndarray:
        if self.df is None:
            raise RuntimeError("load_metadata() must run before build_label_matrix().")

        self.df["diseases_list"] = self.df["Finding Labels"].apply(parse_finding_labels)
        self.df["num_diseases"] = self.df["diseases_list"].apply(len)

        encoder = MultiLabelBinarizer(classes=DISEASE_CLASSES)
        label_matrix = encoder.fit_transform(self.df["diseases_list"])

        if label_matrix.shape[1] != len(DISEASE_CLASSES):
            raise RuntimeError("Label matrix width mismatch with DISEASE_CLASSES.")

        LOGGER.info("Built multi-hot labels with shape %s", label_matrix.shape)
        return label_matrix

    def split_dataframe(
        self,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.df is None or "num_diseases" not in self.df.columns:
            raise RuntimeError("build_label_matrix() must run before split_dataframe().")

        validate_split_ratios(train_ratio, val_ratio, test_ratio)

        stage1_target = safe_stratify_target(self.df["num_diseases"], "Stage-1")
        train_df, temp_df = train_test_split(
            self.df,
            test_size=(val_ratio + test_ratio),
            random_state=self.seed,
            stratify=stage1_target,
        )

        val_ratio_in_temp = val_ratio / (val_ratio + test_ratio)
        stage2_target = safe_stratify_target(temp_df["num_diseases"], "Stage-2")
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_in_temp),
            random_state=self.seed,
            stratify=stage2_target,
        )

        LOGGER.info(
            "Split sizes -> train: %s | val: %s | test: %s",
            f"{len(train_df):,}",
            f"{len(val_df):,}",
            f"{len(test_df):,}",
        )
        return train_df, val_df, test_df

    @staticmethod
    def build_split_dataframe(split_df: pd.DataFrame, label_matrix: np.ndarray, image_lookup: Dict[str, str]) -> pd.DataFrame:
        split_indices = split_df.index.to_numpy()
        split_labels = label_matrix[split_indices]

        out_df = pd.DataFrame(
            {
                "Image Index": split_df["Image Index"].values,
                "image_path": [image_lookup.get(name, "") for name in split_df["Image Index"].values],
                "labels": [row.tolist() for row in split_labels],
            }
        )
        return out_df

    def save_outputs(
        self,
        label_matrix: np.ndarray,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        split_frames = {"train": train_df, "val": val_df, "test": test_df}

        split_json_path = self.processed_data_dir / "data_splits.json"
        with open(split_json_path, "w", encoding="utf-8") as handle:
            json.dump({name: frame["Image Index"].tolist() for name, frame in split_frames.items()}, handle, indent=2)
        LOGGER.info("Saved split index file: %s", split_json_path)

        image_lookup = build_image_lookup(self.raw_data_dir)

        for split_name, split_df in split_frames.items():
            out_df = self.build_split_dataframe(split_df, label_matrix, image_lookup)
            out_csv = self.processed_data_dir / f"{split_name}_labels.csv"
            out_df.to_csv(out_csv, index=False)
            LOGGER.info("Saved %s labels: %s", split_name, out_csv)

        train_indices = train_df.index.to_numpy()
        train_labels = label_matrix[train_indices]
        weights = compute_class_weights(train_labels, DISEASE_CLASSES)

        weights_path = self.processed_data_dir / "class_weights.json"
        with open(weights_path, "w", encoding="utf-8") as handle:
            json.dump(weights, handle, indent=2)
        LOGGER.info("Saved class weights: %s", weights_path)

    def run(self, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
        LOGGER.info("%s", "=" * 72)
        LOGGER.info("Phase 1B | Metadata preprocessing")
        LOGGER.info("%s", "=" * 72)

        self.load_metadata()
        label_matrix = self.build_label_matrix()
        train_df, val_df, test_df = self.split_dataframe(train_ratio, val_ratio, test_ratio)
        self.save_outputs(label_matrix, train_df, val_df, test_df)

        LOGGER.info("Preprocessing complete. Outputs directory: %s", self.processed_data_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare NIH Chest X-ray train/val/test metadata.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path(RAW_DATA_DIR), help="Directory containing Data_Entry_2017.csv and images")
    parser.add_argument("--processed-data-dir", type=Path, default=Path(PROCESSED_DATA_DIR), help="Directory where processed outputs will be written")
    parser.add_argument("--csv-name", type=str, default="Data_Entry_2017.csv", help="Metadata CSV filename inside raw-data-dir")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for data splitting")
    parser.add_argument("--train-split", type=float, default=TRAIN_SPLIT, help="Train split ratio")
    parser.add_argument("--val-split", type=float, default=VAL_SPLIT, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=TEST_SPLIT, help="Test split ratio")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    processor = NIHPreprocessor(
        raw_data_dir=args.raw_data_dir,
        processed_data_dir=args.processed_data_dir,
        csv_name=args.csv_name,
        seed=args.seed,
    )
    processor.run(args.train_split, args.val_split, args.test_split)


if __name__ == "__main__":
    main()
