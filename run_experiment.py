def main():
    """Main function for running experiments"""
    parser = argparse.ArgumentParser(description='Run LLM-SNGAT experiments')
    parser.add_argument('--models', nargs='+', default=['GPT-4o', 'O1-preview', 'DeepSeek-R1'],
                      help='Models to test')
    parser.add_argument('--dataset-path', type=str, default=None,
                      help='Path to AQuA dataset (JSON file, HF directory, or HF dataset name)')
    parser.add_argument('--use-real-llm', action='store_true',
                      help='Use real LLM APIs instead of simulation')
    parser.add_argument('--replications', type=int, default=1,
                      help='Number of experimental replications')
    parser.add_argument('--n-students', type=int, default=150,
                      help='Number of simulated students per group')
    parser.add_argument('--form-size', type=int, default=50,
                      help='Size of each test form')
    parser.add_argument('--common-sizes', nargs='+', type=int, default=[5, 10, 15, 20],
                      help='Common item sizes to test')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Update configuration with command line arguments
    config = Config()
    config.SIMULATION.n_students = args.n_students
    config.SIMULATION.form_size = args.form_size
    config.SIMULATION.common_item_sizes = args.common_sizes
    config.RANDOM_SEED = args.seed
    
    if args.output_dir:
        config.RESULTS_DIR = args.output_dir
        config.FIGURES_DIR = f"{args.output_dir}/figures"
        config.LOGS_DIR = f"{args.output_dir}/logs"
    
    # Initialize experiment runner
    runner = ExperimentRunner(config)
    
    # Check environment
    if not runner.check_environment():
        print("Environment check failed. Please install missing dependencies.")
        sys.exit(1)
    
    print("LLM-SNGAT Experimental Runner")
    print("=" * 50)
    print(f"Models: {args.models}")
    print(f"Dataset: {args.dataset_path or 'Generated sample data'}")
    print(f"Replications: {args.replications}")
    print(f"Students per group: {args.n_students}")
    print(f"Test form size: {args.form_size}")
    print(f"Common item sizes: {args.common_sizes}")
    print(f"Use real LLM: {args.use_real_llm}")
    print(f"Random seed: {args.seed}")
    print("")
    
    # Show dataset information if provided
    if args.dataset_path:
        print("Dataset Information:")
        print("-" * 30)
        
        # Test dataset loading
        try:
            from llm_sngat import AQuaDatasetLoader
            test_loader = AQuaDatasetLoader(args.dataset_path)
            dataset_info = test_loader.get_dataset_info()
            
            print(f"Source: {dataset_info.get('source', 'Unknown')}")
            print(f"Total problems: {dataset_info.get('total_problems', 0)}")
            print(f"Format: {dataset_info.get('format', 'Unknown')}")
            print(f"Avg question length: {dataset_info.get('avg_question_length', 0):.1f} chars")
            
            if 'sample_questions' in dataset_info:
                print("\nSample questions:")
                for i, q in enumerate(dataset_info['sample_questions'], 1):
                    print(f"  {i}. {q}")
            
            # Show dataset analysis
            analysis = test_loader.analyze_dataset()
            print(f"\nDataset Analysis:")
            print(f"  Answer distribution: {analysis.get('answer_distribution', {})}")
            print(f"  Question length range: {analysis.get('question_stats', {}).get('min_length', 0)}-{analysis.get('question_stats', {}).get('max_length', 0)} chars")
            
            print("")
            
        except Exception as e:
            print(f"Warning: Could not load dataset from {args.dataset_path}: {e}")
            print("Will use generated sample data instead.")
            print("")
    
    try:
        # Run comprehensive experiment with dataset path
        all_results = runner.run_comprehensive_experiment(
            models=args.models,
            use_real_llm=args.use_real_llm,
            n_replications=args.replications,
            dataset_path=args.dataset_path,
            seed=args.seed
        )
        
        # Create comparison plots
        comparison_data = runner.create_comparison_plots(all_results)
        
        # Save results
        output_files = runner.save_comprehensive_results(all_results, comparison_data)
        
        # Print summary
        print("\n" + "=" * 50)
        print("EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print("Output files:")
        for file_type, filename in output_files.items():
            if file_type != 'results_df':
                print(f"  {file_type}: {filename}")
        
        # Print quick summary
        results_df = output_files['results_df']
        print(f"\nQuick Summary:")
        print(f"  Total configurations tested: {len(results_df)}")
        print(f"  Models: {results_df['Model'].unique().tolist()}")
        print(f"  Common item sizes: {sorted(results_df['Common_Items'].unique().tolist())}")
        
        best_config = results_df.loc[results_df['Tucker_SE'].idxmin()]
        print(f"  Best configuration: {best_config['Model']} with {best_config['Common_Items']} common items")
        print(f"  Best Tucker SE: {best_config['Tucker_SE']:.4f}")
        
        # Dataset usage summary
        if args.dataset_path:
            print(f"\nDataset Usage:")
            print(f"  Used real AQuA dataset from: {args.dataset_path}")
            print(f"  Problems per test form: {args.form_size}")
            print(f"  Total problems needed: {args.form_size * 2} (minimum)")
        else:
            print(f"\nDataset Usage:")
            print(f"  Used generated sample problems")
            print(f"  To use real AQuA data, download from: https://github.com/google-deepmind/AQuA")
            print(f"  Then run with: --dataset-path /path/to/AQuA/train.json")
        
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nExperiment failed with error: {e}")
        runner.logger.error(f"Experiment failed: {e}", exc_info=True)
        sys.exit(1)#!/usr/bin/env python3
"""
Experimental runner for LLM-SNGAT methodology
This script runs comprehensive experiments across multiple models and configurations
"""

import argparse
import json
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

from config import Config
from utils import (
    setup_logging, save_results, get_timestamp, check_dependencies,
    generate_sample_aqua_problems, format_results_table
)
from llm_sngat import (
    AQuaDatasetLoader, StudentSimulator, LLMResponseSimulator,
    LLMSNGATProcessor, ResultAnalyzer
)

class ExperimentRunner:
    """Main class for running LLM-SNGAT experiments"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.logger = setup_logging()
        self.timestamp = get_timestamp()
        
        # Create output directories
        self.config.create_directories()
        
    def check_environment(self) -> bool:
        """Check if environment is ready for experiments"""
        self.logger.info("Checking environment...")
        
        # Check dependencies
        deps = check_dependencies()
        missing_deps = [pkg for pkg, available in deps.items() if not available]
        
        if missing_deps:
            self.logger.error(f"Missing dependencies: {missing_deps}")
            return False
        
        # Check API keys
        api_status = self.config.validate_api_keys()
        available_models = [model for model, available in api_status.items() if available]
        
        if not available_models:
            self.logger.warning("No API keys found. Will use simulation mode only.")
        else:
            self.logger.info(f"Available models: {available_models}")
        
        return True
    
    def run_single_experiment(self, model_name: str, use_real_llm: bool = False) -> Dict[str, Any]:
        """Run experiment for a single model"""
        self.logger.info(f"Running experiment for {model_name}")
        
        # Initialize components
        dataset_loader = AQuaDatasetLoader()
        
        if use_real_llm and model_name in self.config.MODELS:
            model_config = self.config.get_model_config(model_name)
            simulator = LLMResponseSimulator(model_config.model_id)
        else:
            simulator = LLMResponseSimulator()  # Use simulation mode
        
        processor = LLMSNGATProcessor(dataset_loader, simulator)
        
        # Stage 1: Initial simulation
        self.logger.info("Stage 1: Creating test forms and initial simulation")
        form_x, form_y, students_x, students_y = processor.stage_one_simulation(
            n_students=self.config.SIMULATION.n_students,
            form_size=self.config.SIMULATION.form_size
        )
        
        # Store results for this model
        model_results = {}
        
        # Stage 2: Test different common item sizes
        for n_common in self.config.SIMULATION.common_item_sizes:
            self.logger.info(f"Stage 2: Processing with {n_common} common items")
            
            # Create common items and new forms
            new_form_x, new_form_y, common_x, common_y = processor.stage_two_simulation(
                form_x, form_y, students_x, students_y, n_common
            )
            
            # Calculate scores
            scores_x = processor.calculate_scores(students_x, new_form_x, 'responses_x')
            scores_y = processor.calculate_scores(students_y, new_form_y, 'responses_y')
            
            # Extract common item scores
            common_scores_x = self._extract_common_scores(
                students_x, new_form_x, len(form_x)
            )
            common_scores_y = self._extract_common_scores(
                students_y, new_form_y, len(form_y)
            )
            
            # Perform equating
            equating_results = processor.tucker_levine_equating(
                scores_x, scores_y, common_scores_x, common_scores_y
            )
            
            # Calculate standard errors
            analyzer = ResultAnalyzer()
            standard_errors = analyzer.calculate_standard_errors(
                scores_x, scores_y, equating_results
            )
            
            # Store results
            model_results[n_common] = {
                'tucker_se': standard_errors['tucker_se'],
                'levine_se': standard_errors['levine_se'],
                'scores_x_mean': np.mean(scores_x),
                'scores_y_mean': np.mean(scores_y),
                'scores_x_std': np.std(scores_x),
                'scores_y_std': np.std(scores_y),
                'common_correlation': np.corrcoef(common_scores_x, common_scores_y)[0, 1],
                'equating_results': equating_results