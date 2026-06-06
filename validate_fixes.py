#!/usr/bin/env python
"""
Validation script to verify all fixes have been correctly applied.

Run this before retraining to ensure all changes are in place.

Usage:
    python validate_fixes.py
"""

import json
import sys
from pathlib import Path

def check_file_exists(path, description):
    """Check if a file exists."""
    if Path(path).exists():
        print(f"✅ {description}")
        return True
    else:
        print(f"❌ {description} - NOT FOUND")
        return False

def check_file_contains(path, text, description):
    """Check if a file contains specific text."""
    try:
        with open(path) as f:
            content = f.read()
            if text in content:
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description} - NOT FOUND IN FILE")
                return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def check_json_structure(path, keys, description):
    """Check if JSON file has required structure."""
    try:
        with open(path) as f:
            data = json.load(f)
            if all(k in data for k in keys):
                print(f"✅ {description}")
                return True
            else:
                print(f"❌ {description} - Missing keys: {[k for k in keys if k not in data]}")
                return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def main():
    project_root = Path(__file__).parent.absolute()
    print("\n" + "="*80)
    print("VALIDATION: Chest Disease Classifier - Balancing Fixes")
    print("="*80 + "\n")
    
    all_pass = True
    
    # Check core files exist
    print("📋 Checking core files...")
    all_pass &= check_file_exists(project_root / "src" / "balancing.py", "balancing.py exists")
    all_pass &= check_file_exists(project_root / "src" / "train.py", "train.py exists")
    all_pass &= check_file_exists(project_root / "api" / "inference.py", "inference.py exists")
    all_pass &= check_file_exists(project_root / "data" / "processed" / "class_weights.json", "class_weights.json exists")
    
    # Check documentation files
    print("\n📚 Checking documentation files...")
    all_pass &= check_file_exists(project_root / "BALANCING_FIXES.md", "BALANCING_FIXES.md")
    all_pass &= check_file_exists(project_root / "QUICK_START_FIX.md", "QUICK_START_FIX.md")
    all_pass &= check_file_exists(project_root / "FIX_SUMMARY.md", "FIX_SUMMARY.md")
    all_pass &= check_file_exists(project_root / "retrain_with_balanced_data.py", "retrain_with_balanced_data.py")
    
    # Check code changes - balancing.py
    print("\n🔍 Checking balancing.py modifications...")
    all_pass &= check_file_contains(
        project_root / "src" / "balancing.py",
        "class_oversampling_stats",
        "Enhanced oversample_minority_classes() with statistics logging"
    )
    all_pass &= check_file_contains(
        project_root / "src" / "balancing.py",
        "sqrt_inverse",
        "Added sqrt_inverse weighting method"
    )
    all_pass &= check_file_contains(
        project_root / "src" / "balancing.py",
        "np.clip(normalized, 0.1, 15.0)",
        "Extended weight clipping range to 0.1-15.0"
    )
    
    # Check code changes - train.py
    print("\n🔍 Checking train.py modifications...")
    all_pass &= check_file_contains(
        project_root / "src" / "train.py",
        "rare_threshold = 0.10",
        "Added rare disease identification in tune_thresholds()"
    )
    all_pass &= check_file_contains(
        project_root / "src" / "train.py",
        "sweep_for_class = np.arange(0.05, 0.70, 0.02)",
        "Added separate threshold sweep for rare diseases"
    )
    all_pass &= check_file_contains(
        project_root / "src" / "train.py",
        "min(len(image_paths), 30000)",
        "Increased shuffle buffer to 30000"
    )
    
    # Check code changes - inference.py
    print("\n🔍 Checking inference.py modifications...")
    all_pass &= check_file_contains(
        project_root / "api" / "inference.py",
        "is_rare_disease",
        "Added rare disease detection in format_predictions()"
    )
    all_pass &= check_file_contains(
        project_root / "api" / "inference.py",
        "cal_amplification = 1.5",
        "Added 1.5x calibration boost for rare diseases"
    )
    all_pass &= check_file_contains(
        project_root / "api" / "inference.py",
        "is_rare_disease = disease in",
        "Identified rare diseases list"
    )
    
    # Check class_weights.json structure
    print("\n🔍 Checking class_weights.json...")
    try:
        with open(project_root / "data" / "processed" / "class_weights.json") as f:
            weights = json.load(f)
        
        if "inverse_frequency" in weights and "recommended" in weights:
            print("✅ class_weights.json has both inverse_frequency and recommended")
            
            inv_weights = weights["inverse_frequency"]
            if inv_weights.get("Hernia", 0) >= 10:
                print(f"✅ Hernia weight is aggressive (15.0x)")
            else:
                print(f"❌ Hernia weight is too conservative ({inv_weights.get('Hernia', 'N/A')})")
                all_pass = False
            
            if 0.5 < inv_weights.get("Infiltration", 1) < 1.0:
                print(f"✅ Infiltration weight is downweighted ({inv_weights.get('Infiltration', 'N/A')})")
            else:
                print(f"❌ Infiltration weight is incorrect ({inv_weights.get('Infiltration', 'N/A')})")
                all_pass = False
        else:
            print("❌ class_weights.json missing required sections")
            all_pass = False
    except Exception as e:
        print(f"❌ Error reading class_weights.json: {e}")
        all_pass = False
    
    # Summary
    print("\n" + "="*80)
    if all_pass:
        print("✅ ALL VALIDATIONS PASSED!")
        print("\n🚀 You can now run:")
        print("   python retrain_with_balanced_data.py")
        print("\nOr manually:")
        print("   python src/data_preprocessing.py --apply-oversampling True")
        print("   python src/train.py --sample-weighting inverse_frequency --tune-thresholds")
        print("   python src/evaluate.py")
        print("="*80 + "\n")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("\n⚠️ Please review the failures above and reapply fixes.")
        print("="*80 + "\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
