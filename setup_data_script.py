#!/usr/bin/env python3
"""
Data setup script for LLM-SNGAT project
Downloads and prepares the AQuA dataset
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import argparse

def download_from_github():
    """Download AQuA dataset from GitHub repository"""
    print("Downloading AQuA dataset from GitHub...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    aqua_dir = data_dir / "AQuA"
    
    if aqua_dir.exists():
        print(f"AQuA directory already exists at {aqua_dir}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            return str(aqua_dir)
    
    try:
        # Clone the repository
        cmd = ["git", "clone", "https://github.com/google-deepmind/AQuA.git", str(aqua_dir)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully downloaded AQuA dataset to {aqua_dir}")
            
            # List available files
            files = list(aqua_dir.glob("*.json"))
            print(f"Available files: {[f.name for f in files]}")
            
            return str(aqua_dir)
        else:
            print(f"❌ Error downloading from GitHub: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("❌ Git not found. Please install git first.")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def download_from_huggingface():
    """Download AQuA dataset using Hugging Face datasets"""
    print("Downloading AQuA dataset from Hugging Face...")
    
    try:
        from datasets import load_dataset
        
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        aqua_hf_dir = data_dir / "aqua_rat"
        
        if aqua_hf_dir.exists():
            print(f"Hugging Face AQuA directory already exists at {aqua_hf_dir}")
            response = input("Do you want to re-download? (y/n): ")
            if response.lower() != 'y':
                return str(aqua_hf_dir)
        
        # Download dataset
        print("Loading dataset from Hugging Face hub...")
        dataset = load_dataset('deepmind/aqua_rat', 'raw')
        
        # Save to disk
        dataset.save_to_disk(str(aqua_hf_dir))
        
        print(f"✅ Successfully downloaded AQuA dataset to {aqua_hf_dir}")
        print(f"Dataset splits: {list(dataset.keys())}")
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Test examples: {len(dataset['test'])}")
        print(f"Validation examples: {len(dataset['validation'])}")
        
        return str(aqua_hf_dir)
        
    except ImportError:
        print("❌ Hugging Face datasets not installed.")
        print("Install with: pip install datasets")
        return None
    except Exception as e:
        print(f"❌ Error downloading from Hugging Face: {e}")
        return None

def convert_hf_to_json(hf_path: str):
    """Convert Hugging Face dataset to JSON format"""
    try:
        from datasets import load_from_disk
        
        print(f"Converting Hugging Face dataset to JSON format...")
        
        dataset = load_from_disk(hf_path)
        
        data_dir = Path("data")
        
        # Convert each split to JSON
        for split_name in dataset.keys():
            split_data = dataset[split_name]
            
            json_data = []
            for item in split_data:
                json_data.append({
                    'question': item['question'],
                    'options': item['options'],
                    'rationale': item['rationale'],
                    'correct': item['correct']
                })
            
            json_file = data_dir / f"aqua_{split_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Saved {len(json_data)} problems to {json_file}")
        
        return str(data_dir)
        
    except Exception as e:
        print(f"❌ Error converting to JSON: {e}")
        return None

def verify_dataset(dataset_path: str):
    """Verify that the dataset can be loaded properly"""
    print(f"\nVerifying dataset at {dataset_path}...")
    
    try:
        # Import our dataset loader
        sys.path.append('.')
        from llm_sngat import AQuaDatasetLoader
        
        loader = AQuaDatasetLoader(dataset_path)
        info = loader.get_dataset_info()
        
        print("✅ Dataset verification successful!")
        print(f"Source: {info.get('source', 'Unknown')}")
        print(f"Total problems: {info.get('total_problems', 0)}")
        print(f"Format: {info.get('format', 'Unknown')}")
        print(f"Average question length: {info.get('avg_question_length', 0):.1f} characters")
        
        # Show analysis
        analysis = loader.analyze_dataset()
        if 'answer_distribution' in analysis:
            print(f"Answer distribution: {analysis['answer_distribution']}")
        
        # Test creating forms
        if info.get('total_problems', 0) >= 100:
            print("\nTesting form creation...")
            form_x, form_y = loader.create_test_forms(50)
            print(f"✅ Successfully created test forms: X={len(form_x)}, Y={len(form_y)}")
        else:
            print("⚠️  Dataset has fewer than 100 problems. May not be suitable for large experiments.")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Setup AQuA dataset for LLM-SNGAT')
    parser.add_argument('--method', choices=['github', 'huggingface', 'both'], default='both',
                       help='Download method')
    parser.add_argument('--convert-to-json', action='store_true',
                       help='Convert Hugging Face format to JSON')
    parser.add_argument('--verify-only', type=str,
                       help='Only verify existing dataset at given path')
    
    args = parser.parse_args()
    
    print("AQuA Dataset Setup for LLM-SNGAT")
    print("=" * 40)
    
    if args.verify_only:
        verify_dataset(args.verify_only)
        return
    
    success_paths = []
    
    if args.method in ['github', 'both']:
        github_path = download_from_github()
        if github_path:
            success_paths.append(('GitHub', github_path))
    
    if args.method in ['huggingface', 'both']:
        hf_path = download_from_huggingface()
        if hf_path:
            success_paths.append(('Hugging Face', hf_path))
            
            if args.convert_to_json:
                json_path = convert_hf_to_json(hf_path)
                if json_path:
                    success_paths.append(('JSON Converted', json_path))
    
    print("\n" + "=" * 40)
    print("SETUP SUMMARY")
    print("=" * 40)
    
    if success_paths:
        print("✅ Successfully set up the following datasets:")
        for method, path in success_paths:
            print(f"  {method}: {path}")
            
        print("\nUsage examples:")
        for method, path in success_paths:
            if 'GitHub' in method:
                print(f"  # Using GitHub dataset:")
                print(f"  python llm_sngat.py")
                print(f"  python run_experiment.py --dataset-path {path}/train.json")
            elif 'Hugging Face' in method:
                print(f"  # Using Hugging Face dataset:")
                print(f"  python run_experiment.py --dataset-path {path}")
            elif 'JSON' in method:
                print(f"  # Using converted JSON files:")
                print(f"  python run_experiment.py --dataset-path {path}/aqua_train.json")
        
        print("\nVerifying datasets...")
        for method, path in success_paths:
            print(f"\n{method} Dataset:")
            if 'GitHub' in method:
                verify_dataset(f"{path}/train.json")
            elif 'JSON' in method:
                verify_dataset(f"{path}/aqua_train.json")
            else:
                verify_dataset(path)
                
    else:
        print("❌ No datasets were successfully downloaded.")
        print("\nManual setup options:")
        print("1. Install git and run: git clone https://github.com/google-deepmind/AQuA.git data/AQuA")
        print("2. Install datasets: pip install datasets")
        print("3. Use sample data: python llm_sngat.py (no download required)")
    
    print("\n" + "=" * 40)
    print("NEXT STEPS")
    print("=" * 40)
    print("1. Run basic demo: python llm_sngat.py")
    print("2. Run with real data: python run_experiment.py --dataset-path [PATH]")
    print("3. Explore in Jupyter: jupyter notebook demo.ipynb")
    print("4. Set up LLM APIs for real simulation (optional)")
    print("   export OPENAI_API_KEY='your-key'")
    print("   export DEEPSEEK_API_KEY='your-key'")

if __name__ == "__main__":
    main()
