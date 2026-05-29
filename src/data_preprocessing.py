"""
Phase 1B: Data preprocessing pipeline for NIH Chest X-ray dataset.

Creates:
1) Multi-hot encoded labels for 14 diseases
2) Train/val/test split metadata
3) Per-split label CSV files
4) Class weights JSON for imbalanced training
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DISEASE_CLASSES,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)


class ChestXrayDataProcessor:
    """Preprocess NIH chest X-ray metadata for training."""

    def __init__(self, raw_data_dir=RAW_DATA_DIR, processed_data_dir=PROCESSED_DATA_DIR):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.df = None

    def load_metadata(self):
        csv_path = self.raw_data_dir / "Data_Entry_2017.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        required_cols = {"Image Index", "Finding Labels"}
        missing = required_cols.difference(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        print(f"Loaded metadata: {len(self.df):,} rows")
        return self.df

    @staticmethod
    def parse_diseases(disease_string):
        if pd.isna(disease_string) or disease_string == "No Finding":
            return []
        return [x.strip() for x in str(disease_string).split("|") if x.strip()]

    def build_labels(self):
        if self.df is None:
            raise RuntimeError("Call load_metadata() before build_labels().")

        self.df["diseases_list"] = self.df["Finding Labels"].apply(self.parse_diseases)

        mlb = MultiLabelBinarizer(classes=DISEASE_CLASSES)
        label_matrix = mlb.fit_transform(self.df["diseases_list"])

        if label_matrix.shape[1] != len(DISEASE_CLASSES):
            raise RuntimeError("Label matrix shape mismatch with DISEASE_CLASSES.")

        self.df["num_diseases"] = self.df["diseases_list"].apply(len)
        return label_matrix

    @staticmethod
    def _safe_stratify(series, stage_name):
        counts = series.value_counts()
        min_count = int(counts.min()) if len(counts) else 0

        if min_count < 2:
            print(f"Warning: {stage_name} stratification disabled (least-populated group has {min_count} sample).")
            return None

        return series

    def split_data(self):
        if self.df is None or "num_diseases" not in self.df.columns:
            raise RuntimeError("Call build_labels() before split_data().")

        if not np.isclose(TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT, 1.0):
            raise ValueError("TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must equal 1.0")

        stratify_stage1 = self._safe_stratify(self.df["num_diseases"], "Stage-1")

        train_df, temp_df = train_test_split(
            self.df,
            test_size=(VAL_SPLIT + TEST_SPLIT),
            stratify=stratify_stage1,
            random_state=RANDOM_SEED,
        )

        val_ratio_in_temp = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
        stratify_stage2 = self._safe_stratify(temp_df["num_diseases"], "Stage-2")

        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_in_temp),
            stratify=stratify_stage2,
            random_state=RANDOM_SEED,
        )

        print(f"Split sizes -> train: {len(train_df):,}, val: {len(val_df):,}, test: {len(test_df):,}")
        return train_df, val_df, test_df

    @staticmethod
    def _image_lookup(raw_data_dir):
        lookup = {}
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for path in raw_data_dir.rglob(ext):
                lookup[path.name] = str(path)
        return lookup

    def save_outputs(self, label_matrix, train_df, val_df, test_df):
        splits = {"train": train_df, "val": val_df, "test": test_df}

        with open(self.processed_data_dir / "data_splits.json", "w", encoding="utf-8") as f:
            json.dump({k: v["Image Index"].tolist() for k, v in splits.items()}, f, indent=2)

        image_lookup = self._image_lookup(self.raw_data_dir)

        for split_name, split_df in splits.items():
            indices = split_df.index.to_numpy()
            labels_subset = label_matrix[indices]

            out_df = pd.DataFrame(
                {
                    "Image Index": split_df["Image Index"].values,
                    "image_path": [image_lookup.get(name, "") for name in split_df["Image Index"].values],
                    "labels": [row.tolist() for row in labels_subset],
                }
            )

            out_path = self.processed_data_dir / f"{split_name}_labels.csv"
            out_df.to_csv(out_path, index=False)
            print(f"Saved {split_name} labels: {out_path}")

        train_indices = train_df.index.to_numpy()
        class_weights = self._calculate_class_weights(label_matrix[train_indices])

        with open(self.processed_data_dir / "class_weights.json", "w", encoding="utf-8") as f:
            json.dump(class_weights, f, indent=2)

        print(f"Saved class weights: {self.processed_data_dir / 'class_weights.json'}")

    @staticmethod
    def _calculate_class_weights(train_labels):
        total_samples = len(train_labels)
        num_classes = train_labels.shape[1]

        weights = {}
        for i, disease in enumerate(DISEASE_CLASSES):
            positive_count = int(np.sum(train_labels[:, i]))
            weight = total_samples / (num_classes * max(positive_count, 1))
            weights[disease] = float(weight)

        return weights

    def run(self):
        print("=" * 70)
        print("PHASE 1B: DATA PREPROCESSING")
        print("=" * 70)

        self.load_metadata()
        labels = self.build_labels()
        train_df, val_df, test_df = self.split_data()
        self.save_outputs(labels, train_df, val_df, test_df)

        print("\nPreprocessing complete.")
        print(f"Outputs written to: {self.processed_data_dir}")


if __name__ == "__main__":
    ChestXrayDataProcessor().run()
