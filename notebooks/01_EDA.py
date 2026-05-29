"""
Phase 1A: Exploratory Data Analysis (EDA)
Analyze the NIH Chest X-ray dataset before preprocessing.

This script:
1. Loads dataset metadata from Data_Entry_2017.csv
2. Analyzes disease distribution and co-occurrence
3. Examines image statistics
4. Generates visualizations
5. Provides insights for preprocessing decisions
"""

import os
import sys
from PIL import report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import json

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DISEASE_CLASSES, 
    IMAGE_SIZE, RANDOM_SEED
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

class ChestXrayEDA:
    """Exploratory Data Analysis for Chest X-ray dataset."""
    
    def __init__(self, raw_data_dir=RAW_DATA_DIR, output_dir=PROCESSED_DATA_DIR):
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.csv_path = self.raw_data_dir / "Data_Entry_2017.csv"
        self.df = None
        self.disease_counts = None
        
    def load_dataset(self):
        """Load metadata from CSV file."""
        print("\n" + "="*70)
        print("STEP 1: LOADING DATASET")
        print("="*70)
        
        if not self.csv_path.exists():
            raise FileNotFoundError(f"❌ Dataset not found at {self.csv_path}")
        
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Loaded dataset: {self.csv_path}")
            print(f"   Shape: {self.df.shape}")
            print(f"   Columns: {list(self.df.columns)}")
            return self.df
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            raise
    
    def parse_diseases(self, finding_labels):
        """Parse disease labels from string format."""
        if pd.isna(finding_labels) or finding_labels == "No Finding":
            return []
        return [disease.strip() for disease in str(finding_labels).split("|")]
    
    def analyze_basic_stats(self):
        """Display basic dataset statistics."""
        print("\n" + "="*70)
        print("STEP 2: BASIC STATISTICS")
        print("="*70)
        
        print(f"\n📊 Dataset Info:")
        print(f"   Total images: {len(self.df):,}")
        print(f"   Image column: {self.df.columns[0]}")
        print(f"   Finding column: {self.df.columns[1]}")
        print(f"   Patient ID column: {self.df.columns[2]}")
        
        # Parse diseases
        print(f"\n🔄 Parsing disease labels...")
        self.df['diseases'] = self.df[self.df.columns[1]].apply(self.parse_diseases)
        self.df['num_diseases'] = self.df['diseases'].apply(len)
        
        print(f"   ✅ Disease parsing complete")
        
        # Disease statistics
        print(f"\n📈 Disease Statistics:")
        print(f"   Images with No Finding: {(self.df['num_diseases'] == 0).sum():,}")
        print(f"   Images with 1 disease: {(self.df['num_diseases'] == 1).sum():,}")
        print(f"   Images with 2 diseases: {(self.df['num_diseases'] == 2).sum():,}")
        print(f"   Images with 3+ diseases: {(self.df['num_diseases'] >= 3).sum():,}")
        
        return self.df
    
    def analyze_disease_distribution(self):
        """Analyze distribution of each disease class."""
        print("\n" + "="*70)
        print("STEP 3: DISEASE CLASS DISTRIBUTION")
        print("="*70)
        
        # Count occurrences of each disease
        all_diseases = []
        for disease_list in self.df['diseases']:
            all_diseases.extend(disease_list)
        
        self.disease_counts = Counter(all_diseases)
        
        print(f"\n🦠 Disease Counts (sorted):")
        print("-" * 50)
        
        for i, (disease, count) in enumerate(sorted(self.disease_counts.items(), 
                                                      key=lambda x: x[1], 
                                                      reverse=True), 1):
            percentage = (count / len(self.df)) * 100
            bar = "█" * int(percentage / 2)
            print(f"{i:2d}. {disease:20s} | {count:6d} ({percentage:5.1f}%) {bar}")
        
        return self.disease_counts
    
    def analyze_class_imbalance(self):
        """Analyze class imbalance and calculate weights."""
        print("\n" + "="*70)
        print("STEP 4: CLASS IMBALANCE ANALYSIS")
        print("="*70)
        
        # Calculate class weights for imbalanced dataset
        class_weights = {}
        counts = []
        
        for disease in DISEASE_CLASSES:
            if disease in self.disease_counts:
                count = self.disease_counts[disease]
            else:
                count = 0
            counts.append(count)
        
        total_samples = len(self.df)
        
        print(f"\n⚖️  Class Weights (for handling imbalance):")
        print("-" * 50)
        
        for disease, count in zip(DISEASE_CLASSES, counts):
            if count == 0:
                weight = 0.0
                print(f"   {disease:20s} | Count: {count:6d} | Weight: N/A (not found)")
            else:
                # Weight inversely proportional to frequency
                weight = total_samples / (len(DISEASE_CLASSES) * max(count, 1))
                class_weights[disease] = float(weight)
                ratio = (count / total_samples) * 100
                print(f"   {disease:20s} | Count: {count:6d} ({ratio:5.1f}%) | Weight: {weight:.3f}")
        
        # Save class weights
        weights_path = self.output_dir / "class_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(class_weights, f, indent=2)
        print(f"\n✅ Class weights saved to {weights_path}")
        
        return class_weights
    
    def analyze_cooccurrence(self):
        """Analyze co-occurrence of disease pairs."""
        print("\n" + "="*70)
        print("STEP 5: DISEASE CO-OCCURRENCE ANALYSIS")
        print("="*70)
        
        # Create co-occurrence matrix
        cooccurrence = {}
        
        for disease_list in self.df['diseases']:
            if len(disease_list) >= 2:
                # Count pairs
                for i, d1 in enumerate(disease_list):
                    for d2 in disease_list[i+1:]:
                        pair = tuple(sorted([d1, d2]))
                        cooccurrence[pair] = cooccurrence.get(pair, 0) + 1
        
        # Top co-occurrences
        print(f"\n🔗 Top Disease Pairs:")
        print("-" * 50)
        
        top_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:10]
        for i, (pair, count) in enumerate(top_pairs, 1):
            print(f"{i:2d}. {pair[0]:20s} + {pair[1]:20s} | {count:5d} cases")
        
        return cooccurrence
    
    def generate_visualizations(self):
        """Generate and save visualization plots."""
        print("\n" + "="*70)
        print("STEP 6: GENERATING VISUALIZATIONS")
        print("="*70)
        
        # 1. Disease distribution bar chart
        print("\n📊 Creating disease distribution chart...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        diseases_sorted = sorted(self.disease_counts.items(), 
                                key=lambda x: x[1], reverse=True)
        diseases, counts = zip(*diseases_sorted)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(diseases)))
        bars = ax.barh(diseases, counts, color=colors)
        
        ax.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Disease Distribution in Chest X-ray Dataset', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{int(count):,}', 
                   ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        fig_path = self.output_dir / "01_disease_distribution.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved to {fig_path}")
        plt.close()
        
        # 2. Number of diseases per image distribution
        print("\n📊 Creating number of diseases distribution...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        disease_counts = self.df['num_diseases'].value_counts().sort_index()
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(disease_counts)))
        
        bars = ax.bar(disease_counts.index, disease_counts.values, color=colors, edgecolor='black')
        ax.set_xlabel('Number of Diseases per Image', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Multi-label Cases', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        total = disease_counts.sum()
        for bar, (idx, count) in zip(bars, disease_counts.items()):
            height = bar.get_height()
            percentage = (count / total) * 100
            ax.text(bar.get_x() + bar.get_width()/2, height, 
                   f'{percentage:.1f}%\n({int(count):,})', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        fig_path = self.output_dir / "02_multilabel_distribution.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved to {fig_path}")
        plt.close()
        
        # 3. Class imbalance visualization
        print("\n📊 Creating class imbalance chart...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        counts = [self.disease_counts.get(disease, 0) for disease in DISEASE_CLASSES]
        percentages = [(count / len(self.df)) * 100 for count in counts]
        
        colors = ['#ff9999' if p < 5 else '#99ccff' if p < 10 else '#99ff99' 
                 for p in percentages]
        
        bars = ax.barh(DISEASE_CLASSES, percentages, color=colors, edgecolor='black')
        ax.set_xlabel('Percentage of Dataset (%)', fontsize=12, fontweight='bold')
        ax.set_title('Class Imbalance: Percentage of Each Disease Class', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        # Color legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ff9999', edgecolor='black', label='Rare (<5%)'),
            Patch(facecolor='#99ccff', edgecolor='black', label='Moderate (5-10%)'),
            Patch(facecolor='#99ff99', edgecolor='black', label='Common (>10%)')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        fig_path = self.output_dir / "03_class_imbalance.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved to {fig_path}")
        plt.close()
        
        print("\n✅ All visualizations generated!")
        return True
    
    def generate_summary_report(self):
        """Generate a text summary report."""
        print("\n" + "="*70)
        print("STEP 7: GENERATING SUMMARY REPORT")
        print("="*70)
        
        report = f"""
╔════════════════════════════════════════════════════════════════════════╗
║         CHEST X-RAY DATASET - EXPLORATORY DATA ANALYSIS REPORT         ║
╚════════════════════════════════════════════════════════════════════════╝

📊 DATASET OVERVIEW
─────────────────────────────────────────────────────────────────────────
Total Images:              {len(self.df):,}
Image Format:              PNG (Grayscale)
Disease Classes:           {len(DISEASE_CLASSES)} (multi-label)
Target Size:               {IMAGE_SIZE}x{IMAGE_SIZE} pixels

🔍 LABEL DISTRIBUTION
─────────────────────────────────────────────────────────────────────────
Images with "No Finding":  {(self.df['num_diseases'] == 0).sum():,} ({(self.df['num_diseases'] == 0).sum()/len(self.df)*100:.1f}%)
Images with 1 disease:     {(self.df['num_diseases'] == 1).sum():,} ({(self.df['num_diseases'] == 1).sum()/len(self.df)*100:.1f}%)
Images with 2 diseases:    {(self.df['num_diseases'] == 2).sum():,} ({(self.df['num_diseases'] == 2).sum()/len(self.df)*100:.1f}%)
Images with 3+ diseases:   {(self.df['num_diseases'] >= 3).sum():,} ({(self.df['num_diseases'] >= 3).sum()/len(self.df)*100:.1f}%)

⚖️  CLASS IMBALANCE RATIO
─────────────────────────────────────────────────────────────────────────
Most common disease:       {sorted(self.disease_counts.items(), key=lambda x: x[1], reverse=True)[0][0]:20s} ({sorted(self.disease_counts.items(), key=lambda x: x[1], reverse=True)[0][1]:,} cases)
Rarest disease:            {sorted(self.disease_counts.items(), key=lambda x: x[1])[0][0]:20s} ({sorted(self.disease_counts.items(), key=lambda x: x[1])[0][1]:,} cases)
Imbalance Ratio:           {sorted(self.disease_counts.items(), key=lambda x: x[1], reverse=True)[0][1] / max(sorted(self.disease_counts.items(), key=lambda x: x[1])[0][1], 1):.1f}:1

💡 KEY INSIGHTS
─────────────────────────────────────────────────────────────────────────
✓ Multi-label classification task (images can have multiple diseases)
✓ Significant class imbalance - some diseases are much rarer
✓ "No Finding" is a common class - requires handling
✓ Need class weights for balanced training
✓ Consider stratified train/val/test split

📋 PREPROCESSING PLAN
─────────────────────────────────────────────────────────────────────────
1. Resize all images to {IMAGE_SIZE}x{IMAGE_SIZE}
2. Normalize pixel values to [0, 1]
3. Create multi-hot encoded labels for 14 diseases
4. Calculate class weights for imbalance handling
5. Split into train ({80:.0f}%), validation ({10:.0f}%), test ({10:.0f}%)
6. Use stratified split based on number of diseases

🚀 NEXT STEPS
─────────────────────────────────────────────────────────────────────────
1. Run: python src/data_preprocessing.py
2. Creates train/val/test splits
3. Generates label files and metadata
4. Proceed to Phase 2: Model Training
"""
        
        print(report)
        
        # Save report
        report_path = self.output_dir / "EDA_REPORT.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n✅ Report saved to {report_path}")
    
    def run_full_eda(self):
        """Run complete EDA pipeline."""
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " "*68 + "║")
        print("║" + "  PHASE 1A: EXPLORATORY DATA ANALYSIS (EDA)".center(68) + "║")
        print("║" + "  Chest Disease Classifier Project".center(68) + "║")
        print("║" + " "*68 + "║")
        print("╚" + "="*68 + "╝")
        
        try:
            self.load_dataset()
            self.analyze_basic_stats()
            self.analyze_disease_distribution()
            self.analyze_class_imbalance()
            self.analyze_cooccurrence()
            self.generate_visualizations()
            self.generate_summary_report()
            
            print("\n" + "="*70)
            print("✅ EDA COMPLETE!")
            print("="*70)
            print(f"\n📁 Output files saved to: {self.output_dir}")
            print(f"\n🚀 Next step: python src/data_preprocessing.py\n")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error during EDA: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Run EDA
    eda = ChestXrayEDA()
    success = eda.run_full_eda()
    
    # Exit with appropriate code
    exit(0 if success else 1)