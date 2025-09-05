#!/usr/bin/env python3
"""
Simple runner script for stroke LLM evaluation
"""

import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config import APIConfig, EVALUATION_CONFIG, MODEL_CONFIGS, OUTPUT_DIRS
from stroke_llm_evaluation import (
    DataPreprocessor, LLMEvaluator, BaselineEvaluator, 
    StatisticalAnalyzer, Visualizer, ModelConfig
)

def setup_directories():
    """Create necessary output directories"""
    for dir_name in OUTPUT_DIRS.values():
        os.makedirs(dir_name, exist_ok=True)

def setup_logging():
    """Setup logging configuration"""
    log_file = f"logs/evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def create_model_configs():
    """Create model configurations from config file"""
    api_config = APIConfig()
    
    model_configs = {
        'gpt-4o': ModelConfig('gpt-4o', api_config.OPENAI_API_KEY),
        'o3': ModelConfig('o3', api_config.OPENAI_API_KEY),
        'claude-opus-4': ModelConfig('claude-opus-4', api_config.ANTHROPIC_API_KEY),
        'claude-sonnet-4': ModelConfig('claude-sonnet-4', api_config.ANTHROPIC_API_KEY),
        'gemini-2.5-pro': ModelConfig('gemini-2.5-pro', api_config.GOOGLE_API_KEY),
        'gemini-2.5-flash': ModelConfig('gemini-2.5-flash', api_config.GOOGLE_API_KEY),
    }
    
    # Add other models if API keys are provided
    if api_config.GROK_API_KEY and api_config.GROK_API_KEY != 'your-grok-api-key-here':
        model_configs['grok-3-beta'] = ModelConfig(
            'grok-3-beta', 
            api_config.GROK_API_KEY, 
            'https://api.x.ai/v1/chat/completions'
        )
    
    return model_configs

def run_baseline_evaluation(data_processor, logger):
    """Run baseline model evaluation"""
    logger.info("Starting baseline evaluation...")
    
    df = data_processor.load_data()
    X, y_regression, y_classification = data_processor.preprocess_data(df)
    
    baseline_evaluator = BaselineEvaluator()
    baseline_results = baseline_evaluator.evaluate_baselines(X, y_regression, y_classification)
    
    logger.info("Baseline evaluation completed")
    return baseline_results, df

def run_llm_evaluation(model_configs, df, logger):
    """Run LLM evaluation"""
    logger.info("Starting LLM evaluation...")
    
    llm_evaluator = LLMEvaluator(model_configs)
    llm_results = []
    
    # Use subset for testing to manage API costs
    df_sample = df.sample(n=min(EVALUATION_CONFIG['test_sample_size'], len(df)), random_state=42)
    
    # Start with subset of models and strategies for testing
    test_models = ['gpt-4o', 'claude-sonnet-4']  # Add more as needed
    test_strategies = ['zero_shot', 'cot']  # Add 'few_shot_3' after initial testing
    
    for model_name in test_models:
        if model_name in model_configs:
            for task in ['recovery', 'classification']:
                for strategy in test_strategies:
                    try:
                        logger.info(f"Evaluating {model_name} on {task} with {strategy}")
                        result = llm_evaluator.evaluate_model(
                            model_name, df_sample, task, strategy
                        )
                        llm_results.append(result)
                    except Exception as e:
                        logger.error(f"Error evaluating {model_name} on {task} with {strategy}: {e}")
        else:
            logger.warning(f"Model {model_name} not configured (missing API key)")
    
    logger.info("LLM evaluation completed")
    return llm_results

def save_results(baseline_results, llm_results, logger):
    """Save evaluation results"""
    logger.info("Saving results...")
    
    # Save LLM results
    if llm_results:
        import pandas as pd
        llm_df = pd.DataFrame([
            {
                'Model': r.model_name,
                'Task': r.task,
                'Strategy': r.prompt_strategy,
                'Performance': r.performance,
                **r.additional_metrics
            }
            for r in llm_results
        ])
        llm_df.to_csv('results/llm_evaluation_results.csv', index=False)
        logger.info("LLM results saved to results/llm_evaluation_results.csv")
    
    # Save baseline results
    if baseline_results:
        import pandas as pd
        baseline_df = pd.DataFrame(baseline_results).T
        baseline_df.to_csv('results/baseline_evaluation_results.csv')
        logger.info("Baseline results saved to results/baseline_evaluation_results.csv")

def create_visualizations(baseline_results, llm_results, logger):
    """Create visualizations"""
    if not llm_results:
        logger.warning("No LLM results to visualize")
        return
    
    logger.info("Creating visualizations...")
    visualizer = Visualizer()
    
    try:
        visualizer.plot_model_comparison(llm_results, baseline_results)
        logger.info("Model comparison plot saved")
        
        visualizer.plot_cross_task_correlation(llm_results)
        logger.info("Cross-task correlation plot saved")
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")

def main():
    """Main execution function"""
    # Setup
    setup_directories()
    logger = setup_logging()
    
    logger.info("Starting Stroke LLM Evaluation")
    logger.info(f"Configuration: {EVALUATION_CONFIG}")
    
    # Initialize data processor
    data_processor = DataPreprocessor('浦沿卒中随访患者结果.xls')
    
    # Create model configurations
    model_configs = create_model_configs()
    logger.info(f"Configured models: {list(model_configs.keys())}")
    
    try:
        # Run baseline evaluation
        baseline_results, df = run_baseline_evaluation(data_processor, logger)
        
        # Run LLM evaluation
        llm_results = run_llm_evaluation(model_configs, df, logger)
        
        # Statistical analysis
        if llm_results:
            analyzer = StatisticalAnalyzer()
            statistical_results = analyzer.compare_models(llm_results, baseline_results)
            logger.info("Statistical analysis completed")
        
        # Save results
        save_results(baseline_results, llm_results, logger)
        
        # Create visualizations
        create_visualizations(baseline_results, llm_results, logger)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Dataset size: {len(df)}")
        print(f"LLM evaluations completed: {len(llm_results)}")
        print(f"Baseline models evaluated: {len(baseline_results)}")
        
        if llm_results:
            best_llm_recovery = max([r for r in llm_results if r.task == 'recovery'], 
                                  key=lambda x: x.performance)
            print(f"Best LLM (Recovery): {best_llm_recovery.model_name} ({best_llm_recovery.prompt_strategy}) - {best_llm_recovery.performance:.4f}")
            
            best_llm_classification = max([r for r in llm_results if r.task == 'classification'], 
                                        key=lambda x: x.performance)
            print(f"Best LLM (Classification): {best_llm_classification.model_name} ({best_llm_classification.prompt_strategy}) - {best_llm_classification.performance:.4f}")
        
        print("\nResults saved to 'results/' directory")
        print("Visualizations saved to current directory")
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
