"""
Configuration file for Stroke LLM Evaluation Project
"""

import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class APIConfig:
    """API configuration for different LLM providers"""
    
    # OpenAI API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')
    
    # Anthropic API Key
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', 'your-anthropic-api-key-here')
    
    # Google API Key
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'your-google-api-key-here')
    
    # Other model API keys
    GROK_API_KEY = os.getenv('GROK_API_KEY', 'your-grok-api-key-here')
    LLAMA_API_KEY = os.getenv('LLAMA_API_KEY', 'your-llama-api-key-here')
    QWEN_API_KEY = os.getenv('QWEN_API_KEY', 'your-qwen-api-key-here')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'your-deepseek-api-key-here')

# Model endpoints configuration
MODEL_ENDPOINTS = {
    'grok-3-beta': 'https://api.x.ai/v1/chat/completions',
    'llama-4-maverick': 'https://api.together.xyz/v1/chat/completions',
    'qwen-2.5-max': 'https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation',
    'deepseek-r1': 'https://api.deepseek.com/v1/chat/completions'
}

# Evaluation settings
EVALUATION_CONFIG = {
    'n_runs': 3,                    # Number of runs per evaluation
    'sample_size': 152,             # Full dataset size
    'test_sample_size': 50,         # Subset for LLM testing (to manage API costs)
    'random_state': 42,
    'rate_limit_delay': 0.1,        # Seconds between API calls
    'max_retries': 3
}

# Model-specific configurations
MODEL_CONFIGS = {
    'gpt-4o-2025-03-26': {
        'max_tokens': 8192,
        'temperature': 0.0,
        'top_p': 0.7
    },
    'o3-2025-04-16': {
        'max_tokens': 8192,
        'temperature': 0.0,
        'top_p': 0.7
    },
    'claude-opus-4-20250514': {
        'max_tokens': 8192,
        'temperature': 0.0
    },
    'claude-sonnet-4-20250514': {
        'max_tokens': 8192,
        'temperature': 0.0
    },
    'gemini-2.5-pro-preview-2024-06': {
        'max_tokens': 8192,
        'temperature': 0.0
    },
    'gemini-2.5-flash-preview-2024-05': {
        'max_tokens': 8192,
        'temperature': 0.0
    }
}

# Prompt templates directory
PROMPT_TEMPLATES_DIR = 'prompts'

# Output directories
OUTPUT_DIRS = {
    'results': 'results',
    'plots': 'plots', 
    'logs': 'logs',
    'data': 'data'
}

# Data file paths
DATA_PATHS = {
    'raw_data': '浦沿卒中随访患者结果.xls',
    'processed_data': 'data/processed_stroke_data.csv',
    'few_shot_examples': 'data/few_shot_examples.json'
}
