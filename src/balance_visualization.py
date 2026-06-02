from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def generate_balance_summary(balance_reports: Dict[str, Dict], output_path: Optional[Path] = None) -> str:
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("DATASET CLASS BALANCE ANALYSIS")
    summary_lines.append("=" * 80)
    
    for split_name, report in balance_reports.items():
        summary_lines.append(f"\n{split_name.upper()} SPLIT")
        summary_lines.append("-" * 80)
        summary_lines.append(f"Total Samples: {report['total_samples']:}")
        summary_lines.append(f"Healthy Samples: {report['healthy_samples']:} ({report['healthy_percentage']:.2f}%)")
        summary_lines.append(f"Diseased Samples: {report['diseased_samples']:} ({100 - report['healthy_percentage']:.2f}%)")
        summary_lines.append(f"Imbalance Ratio: {report['imbalance_ratio']:.2f}x")
        summary_lines.append(f"\nClass Distribution:")
        
        class_dist = report["class_distribution"]
        sorted_classes = sorted(class_dist.items(), key=lambda x: x[1]["count"], reverse=True)
        
        for disease, stats in sorted_classes:
            count = stats["count"]
            percentage = stats["percentage"]
            bar_length = int(percentage / 2)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            summary_lines.append(f"  {disease:25} | {count:6} ({percentage:6.2f}%) | {bar}")
    
    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("RECOMMENDATIONS FOR HANDLING CLASS IMBALANCE:")
    summary_lines.append("=" * 80)
    summary_lines.append("- oversampling in preprocessing")
    summary_lines.append("- sample weights in training")
    summary_lines.append("- extra augmentation on minority classes")
    summary_lines.append("- weighted AUC in evaluation")
    summary_lines.append("- fixed threshold at inference")
    summary_lines.append("")
    
    summary_text = "\n".join(summary_lines)
    
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
    
    return summary_text


def compare_class_frequencies(
    train_report: Dict,
    val_report: Dict,
    test_report: Dict,
) -> Dict[str, Dict[str, float]]:
    comparison = {}
    
    train_dist = train_report["class_distribution"]
    val_dist = val_report["class_distribution"]
    test_dist = test_report["class_distribution"]
    
    for disease in train_dist.keys():
        train_pct = train_dist[disease]["percentage"]
        val_pct = val_dist[disease]["percentage"]
        test_pct = test_dist[disease]["percentage"]
        
        comparison[disease] = {
            "train_percentage": train_pct,
            "val_percentage": val_pct,
            "test_percentage": test_pct,
            "std_dev": float(np.std([train_pct, val_pct, test_pct])),
        }
    
    return comparison


def identify_critical_classes(
    balance_report: Dict,
    min_samples_threshold: int = 100,
    max_frequency_threshold: float = 0.05,
) -> List[str]:
    critical = []
    class_dist = balance_report["class_distribution"]
    
    for disease, stats in class_dist.items():
        count = stats["count"]
        percentage = stats["percentage"]
        
        if count < min_samples_threshold or percentage < max_frequency_threshold:
            critical.append(disease)
    
    return critical


def generate_weight_adjustment_report(
    class_weights: Dict[str, float],
    class_distribution: Dict[str, Dict],
) -> Dict[str, object]:
    report = {
        "total_classes": len(class_weights),
        "weight_statistics": {
            "min_weight": float(min(class_weights.values())),
            "max_weight": float(max(class_weights.values())),
            "mean_weight": float(np.mean(list(class_weights.values()))),
            "std_weight": float(np.std(list(class_weights.values()))),
        },
        "class_adjustments": {}
    }
    
    for disease, weight in class_weights.items():
        if disease in class_distribution:
            frequency = class_distribution[disease]["percentage"]
            report["class_adjustments"][disease] = {
                "frequency_percentage": frequency,
                "weight": weight,
                "adjustment_factor": weight / float(np.mean(list(class_weights.values()))),
            }
    
    return report


def print_balance_summary(balance_reports: Dict[str, Dict]) -> None:
    summary = generate_balance_summary(balance_reports)
    print(summary)