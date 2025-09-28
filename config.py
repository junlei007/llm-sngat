"""
Configuration file for LLM-SNGAT project
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for different LLM models"""
    name: str
    api_key_env: str
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 500
    
@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    n_students: int = 150
    form_size: int = 50
    math_ability_mean: float = 75.0
    math_ability_std: float = 10.0
    math_ability_max: float = 100.0
    common_item_sizes: List[int] = None
    
    def __post_init__(self):
        if self.common_item_sizes is None:
            self.common_item_sizes = [5, 10, 15, 20]

@dataclass
class PlotConfig:
    """Configuration for plotting styles"""
    figure_size: tuple = (10, 7)
    detailed_figure_size: tuple = (15, 12)
    dpi: int = 600
    font_size: int = 14
    title_font_size: int = 18
    label_font_size: int = 16
    legend_font_size: int = 13
    line_width: float = 2.5
    marker_size: int = 7
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'GPT-4o': '#1f4e79',
                'O1-preview': '#c5504b',
                'DeepSeek-R1': '#70ad47',
                'Claude': '#8B4A9C',
                'Gemini': '#FF6B35'
            }

class Config:
    """Main configuration class"""
    
    # Project paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
    FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')
    LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
    
    # Dataset configuration
    AQUA_DATASET_PATH = os.path.join(DATA_DIR, 'aqua_dataset.json')
    SAMPLE_PROBLEMS_PATH = os.path.join(DATA_DIR, 'sample_problems.json')
    
    # Model configurations
    MODELS = {
        'GPT-4o': ModelConfig(
            name='GPT-4o',
            api_key_env='OPENAI_API_KEY',
            model_id='gpt-4o',
            temperature=0.7,
            max_tokens=500
        ),
        'O1-preview': ModelConfig(
            name='O1-preview',
            api_key_env='OPENAI_API_KEY',
            model_id='o1-preview',
            temperature=0.0,  # O1 doesn't support temperature
            max_tokens=1000
        ),
        'DeepSeek-R1': ModelConfig(
            name='DeepSeek-R1',
            api_key_env='DEEPSEEK_API_KEY',
            model_id='deepseek-r1',
            temperature=0.7,
            max_tokens=500
        ),
        'Claude': ModelConfig(
            name='Claude',
            api_key_env='ANTHROPIC_API_KEY',
            model_id='claude-3-sonnet-20240229',
            temperature=0.7,
            max_tokens=500
        )
    }
    
    # Simulation configuration
    SIMULATION = SimulationConfig()
    
    # Plot configuration
    PLOT = PlotConfig()
    
    # Logging configuration
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # File naming patterns
    RESULTS_FILENAME_PATTERN = 'llm_sngat_results_{timestamp}.xlsx'
    COMPARISON_PLOT_FILENAME = 'model_comparison_{timestamp}.png'
    DETAILED_PLOT_FILENAME = 'detailed_analysis_{timestamp}.png'
    
    # Equating method configurations
    EQUATING_METHODS = ['tucker', 'levine']
    
    # Validation settings
    RANDOM_SEED = 42
    N_BOOTSTRAP_SAMPLES = 1000
    CONFIDENCE_LEVEL = 0.95
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        for directory in [cls.DATA_DIR, cls.RESULTS_DIR, cls.FIGURES_DIR, cls.LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(cls.MODELS.keys())}")
        return cls.MODELS[model_name]
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """Check which API keys are available"""
        api_status = {}
        for model_name, config in cls.MODELS.items():
            api_key = os.getenv(config.api_key_env)
            api_status[model_name] = api_key is not None and api_key.strip() != ""
        return api_status
