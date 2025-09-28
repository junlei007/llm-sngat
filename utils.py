"""
Utility functions for LLM-SNGAT project
"""

import logging
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import matplotlib.pyplot as plt

from config import Config

def setup_logging(log_level: str = None, log_file: str = None):
    """Setup logging configuration"""
    if log_level is None:
        log_level = Config.LOG_LEVEL
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{Config.LOGS_DIR}/llm_sngat_{timestamp}.log"
    
    Config.create_directories()
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=Config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def save_results(data: Any, filename: str, format: str = 'json'):
    """Save results to file in specified format"""
    Config.create_directories()
    
    if format == 'json':
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    elif format == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'csv' and isinstance(data, pd.DataFrame):
        data.to_csv(filename, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_results(filename: str, format: str = 'json'):
    """Load results from file"""
    if format == 'json':
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif format == 'pickle':
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif format == 'csv':
        return pd.read_csv(filename)
    else:
        raise ValueError(f"Unsupported format: {format}")

def calculate_descriptive_statistics(data: List[float]) -> Dict[str, float]:
    """Calculate comprehensive descriptive statistics"""
    data_array = np.array(data)
    
    stats = {
        'count': len(data),
        'mean': np.mean(data_array),
        'std': np.std(data_array, ddof=1),
        'min': np.min(data_array),
        'max': np.max(data_array),
        'median': np.median(data_array),
        'q25': np.percentile(data_array, 25),
        'q75': np.percentile(data_array, 75),
        'iqr': np.percentile(data_array, 75) - np.percentile(data_array, 25),
        'skewness': calculate_skewness(data_array),
        'kurtosis': calculate_kurtosis(data_array)
    }
    
    return stats

def calculate_skewness(data: np.ndarray) -> float:
    """Calculate sample skewness"""
    n = len(data)
    if n < 3:
        return np.nan
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        return np.nan
    
    skew = np.sum(((data - mean) / std) ** 3) * n / ((n - 1) * (n - 2))
    return skew

def calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate sample excess kurtosis"""
    n = len(data)
    if n < 4:
        return np.nan
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    if std == 0:
        return np.nan
    
    # Sample excess kurtosis
    kurt = (np.sum(((data - mean) / std) ** 4) * n * (n + 1) / 
            ((n - 1) * (n - 2) * (n - 3))) - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    
    return kurt

def bootstrap_confidence_interval(data: List[float], statistic_func, 
                                n_bootstrap: int = 1000, 
                                confidence_level: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for a statistic"""
    np.random.seed(Config.RANDOM_SEED)
    
    data_array = np.array(data)
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data_array, size=len(data_array), replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    
    return ci_lower, ci_upper

def validate_test_forms(form_x: List[Dict], form_y: List[Dict]) -> Dict[str, bool]:
    """Validate test forms for common issues"""
    validation_results = {
        'forms_non_empty': len(form_x) > 0 and len(form_y) > 0,
        'forms_equal_length': len(form_x) == len(form_y),
        'no_duplicates_within_x': len(set(item['question'] for item in form_x)) == len(form_x),
        'no_duplicates_within_y': len(set(item['question'] for item in form_y)) == len(form_y),
        'no_overlap_between_forms': len(set(item['question'] for item in form_x) & 
                                        set(item['question'] for item in form_y)) == 0,
        'all_have_correct_answers': all('correct' in item for item in form_x + form_y),
        'all_have_options': all(len(item.get('options', [])) == 5 for item in form_x + form_y)
    }
    
    return validation_results

def calculate_reliability_coefficient(scores: List[int], form_length: int) -> float:
    """Calculate Cronbach's alpha reliability coefficient approximation"""
    if len(scores) < 2:
        return np.nan
    
    # Simplified reliability calculation
    # In practice, would need item-level data for true Cronbach's alpha
    score_variance = np.var(scores, ddof=1)
    max_possible_variance = (form_length ** 2) / 4  # Maximum variance for binary items
    
    # Approximate reliability using score variance
    reliability = max(0, 1 - (form_length * (1 - score_variance / max_possible_variance)))
    return min(1, reliability)

def format_results_table(results_df: pd.DataFrame) -> str:
    """Format results DataFrame as a nice table string"""
    table_str = "\n" + "="*80 + "\n"
    table_str += "LLM-SNGAT RESULTS SUMMARY\n"
    table_str += "="*80 + "\n"
    
    # Group by model and show summary statistics
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        table_str += f"\n{model}:\n"
        table_str += "-" * (len(model) + 1) + "\n"
        
        for _, row in model_data.iterrows():
            table_str += f"  Common Items: {row['Common_Items']:2d} "
            table_str += f"({row['Anchor_Proportion']:>4s}) | "
            table_str += f"Tucker SE: {row['Tucker_SE']:.4f} | "
            table_str += f"Levine SE: {row['Levine_SE']:.4f}\n"
    
    # Overall statistics
    table_str += "\n" + "="*40 + "\n"
    table_str += "OVERALL STATISTICS\n"
    table_str += "="*40 + "\n"
    
    overall_stats = results_df.groupby('Model').agg({
        'Tucker_SE': ['mean', 'std', 'min', 'max'],
        'Levine_SE': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    table_str += str(overall_stats)
    table_str += "\n" + "="*80 + "\n"
    
    return table_str

def generate_sample_aqua_problems(n_problems: int = 100) -> List[Dict[str, Any]]:
    """Generate sample AQua-style math problems for testing"""
    problems = []
    
    # Problem templates
    templates = [
        {
            'type': 'arithmetic',
            'operations': ['+', '-', '*', '/']
        },
        {
            'type': 'percentage',
            'patterns': ['percent_of', 'increase_decrease', 'find_percentage']
        },
        {
            'type': 'algebra',
            'patterns': ['linear_equation', 'quadratic', 'system']
        },
        {
            'type': 'geometry',
            'patterns': ['area', 'perimeter', 'volume']
        },
        {
            'type': 'word_problem',
            'patterns': ['rate_time_distance', 'work_rate', 'mixture']
        }
    ]
    
    np.random.seed(Config.RANDOM_SEED)
    
    for i in range(n_problems):
        template = np.random.choice(templates)
        problem = generate_problem_by_template(template, i)
        problems.append(problem)
    
    return problems

def generate_problem_by_template(template: Dict, problem_id: int) -> Dict[str, Any]:
    """Generate a single problem based on template"""
    problem_type = template['type']
    
    if problem_type == 'arithmetic':
        return generate_arithmetic_problem(problem_id)
    elif problem_type == 'percentage':
        return generate_percentage_problem(problem_id)
    elif problem_type == 'algebra':
        return generate_algebra_problem(problem_id)
    elif problem_type == 'geometry':
        return generate_geometry_problem(problem_id)
    else:  # word_problem
        return generate_word_problem(problem_id)

def generate_arithmetic_problem(problem_id: int) -> Dict[str, Any]:
    """Generate arithmetic problem"""
    a = np.random.randint(10, 100)
    b = np.random.randint(1, 50)
    operation = np.random.choice(['+', '-', '*'])
    
    if operation == '+':
        answer = a + b
        question = f"What is {a} + {b}?"
    elif operation == '-':
        answer = a - b
        question = f"What is {a} - {b}?"
    else:  # multiplication
        answer = a * b
        question = f"What is {a} × {b}?"
    
    # Generate distractors
    distractors = []
    for _ in range(4):
        distractor = answer + np.random.randint(-20, 21)
        if distractor != answer and distractor not in distractors:
            distractors.append(distractor)
    
    while len(distractors) < 4:
        distractor = answer + np.random.randint(-50, 51)
        if distractor != answer and distractor not in distractors:
            distractors.append(distractor)
    
    # Create options
    all_options = [answer] + distractors[:4]
    np.random.shuffle(all_options)
    
    correct_letter = chr(65 + all_options.index(answer))  # A, B, C, D, E
    options = [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(all_options)]
    
    return {
        'id': problem_id,
        'question': question,
        'options': options,
        'correct': correct_letter,
        'type': 'arithmetic',
        'difficulty': 'easy'
    }

def generate_percentage_problem(problem_id: int) -> Dict[str, Any]:
    """Generate percentage problem"""
    percentage = np.random.choice([10, 15, 20, 25, 30, 40, 50, 60, 75, 80])
    number = np.random.randint(20, 200)
    
    answer = (percentage * number) // 100
    question = f"What is {percentage}% of {number}?"
    
    # Generate distractors
    distractors = [
        (percentage * number) // 10,  # Common error: forgot to divide by 100
        percentage + number,  # Addition instead of percentage
        number - percentage,  # Subtraction
        (number * 100) // percentage  # Inverted calculation
    ]
    
    # Remove duplicates and ensure we have enough distractors
    distractors = [d for d in distractors if d != answer]
    while len(distractors) < 4:
        distractors.append(answer + np.random.randint(-10, 11))
    
    # Create options
    all_options = [answer] + distractors[:4]
    np.random.shuffle(all_options)
    
    correct_letter = chr(65 + all_options.index(answer))
    options = [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(all_options)]
    
    return {
        'id': problem_id,
        'question': question,
        'options': options,
        'correct': correct_letter,
        'type': 'percentage',
        'difficulty': 'medium'
    }

def generate_algebra_problem(problem_id: int) -> Dict[str, Any]:
    """Generate simple algebra problem"""
    x_value = np.random.randint(1, 20)
    constant = np.random.randint(1, 30)
    result = x_value + constant
    
    question = f"If x + {constant} = {result}, what is the value of x?"
    answer = x_value
    
    # Generate distractors
    distractors = [
        result - x_value,  # Used wrong operation
        result + constant,  # Added instead of subtracted
        constant,  # Used constant as answer
        result  # Used result as answer
    ]
    
    distractors = [d for d in distractors if d != answer]
    while len(distractors) < 4:
        distractors.append(answer + np.random.randint(-5, 6))
    
    # Create options
    all_options = [answer] + distractors[:4]
    np.random.shuffle(all_options)
    
    correct_letter = chr(65 + all_options.index(answer))
    options = [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(all_options)]
    
    return {
        'id': problem_id,
        'question': question,
        'options': options,
        'correct': correct_letter,
        'type': 'algebra',
        'difficulty': 'medium'
    }

def generate_geometry_problem(problem_id: int) -> Dict[str, Any]:
    """Generate geometry problem"""
    length = np.random.randint(5, 20)
    width = np.random.randint(3, 15)
    
    area = length * width
    question = f"What is the area of a rectangle with length {length} and width {width}?"
    answer = area
    
    # Generate distractors
    distractors = [
        length + width,  # Perimeter instead of area
        2 * (length + width),  # Full perimeter
        length * width * 2,  # Doubled area
        length + width + length + width  # Another perimeter variant
    ]
    
    distractors = [d for d in distractors if d != answer]
    while len(distractors) < 4:
        distractors.append(answer + np.random.randint(-10, 11))
    
    # Create options
    all_options = [answer] + distractors[:4]
    np.random.shuffle(all_options)
    
    correct_letter = chr(65 + all_options.index(answer))
    options = [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(all_options)]
    
    return {
        'id': problem_id,
        'question': question,
        'options': options,
        'correct': correct_letter,
        'type': 'geometry',
        'difficulty': 'medium'
    }

def generate_word_problem(problem_id: int) -> Dict[str, Any]:
    """Generate word problem"""
    speed = np.random.randint(40, 80)
    time = np.random.randint(2, 8)
    distance = speed * time
    
    question = f"A car travels at {speed} km/h for {time} hours. How far does it travel?"
    answer = distance
    
    # Generate distractors
    distractors = [
        speed + time,  # Addition instead of multiplication
        speed - time,  # Subtraction
        distance // 2,  # Half distance
        distance * 2   # Double distance
    ]
    
    distractors = [d for d in distractors if d != answer and d > 0]
    while len(distractors) < 4:
        distractor = answer + np.random.randint(-50, 51)
        if distractor > 0 and distractor != answer and distractor not in distractors:
            distractors.append(distractor)
    
    # Create options
    all_options = [answer] + distractors[:4]
    np.random.shuffle(all_options)
    
    correct_letter = chr(65 + all_options.index(answer))
    options = [f"{chr(65 + i)}) {opt} km" for i, opt in enumerate(all_options)]
    
    return {
        'id': problem_id,
        'question': question,
        'options': options,
        'correct': correct_letter,
        'type': 'word_problem',
        'difficulty': 'hard'
    }

def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are available"""
    dependencies = {
        'numpy': False,
        'pandas': False,
        'matplotlib': False,
        'seaborn': False,
        'scipy': False,
        'sklearn': False,
        'faker': False,
        'openai': False,
        'openpyxl': False
    }
    
    for package in dependencies:
        try:
            __import__(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    
    return dependencies
