"""
Prompt Loader for Stroke LLM Evaluation
=======================================

This module provides utilities for loading and managing prompt templates
for the stroke neurology LLM evaluation framework.
"""

import json
import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PromptLoader:
    """Load and manage prompt templates for stroke LLM evaluation"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.config = self._load_config()
        self.examples = self._load_examples()
        self._validate_prompts_directory()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load prompt configuration from YAML file"""
        config_path = self.prompts_dir / "prompt_config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {config_path}")
            return {}
    
    def _load_examples(self) -> Dict[str, List[Dict]]:
        """Load few-shot examples from JSON file"""
        examples_path = self.prompts_dir / "few_shot_examples.json"
        if examples_path.exists():
            with open(examples_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.warning(f"Examples file not found: {examples_path}")
            return {}
    
    def _validate_prompts_directory(self):
        """Validate that all required prompt files exist"""
        required_files = [
            "recovery_zero_shot.txt",
            "recovery_cot.txt", 
            "recovery_few_shot.txt",
            "classification_zero_shot.txt",
            "classification_cot.txt",
            "classification_few_shot.txt",
            "few_shot_examples.json",
            "prompt_config.yaml"
        ]
        
        missing_files = []
        for file_name in required_files:
            file_path = self.prompts_dir / file_name
            if not file_path.exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.warning(f"Missing prompt files: {missing_files}")
    
    def load_prompt_template(self, task: str, strategy: str) -> str:
        """
        Load a specific prompt template
        
        Args:
            task: 'recovery_prediction' or 'stroke_classification'
            strategy: 'zero_shot', 'chain_of_thought', or 'few_shot'
            
        Returns:
            Prompt template string
        """
        # Map task names
        task_mapping = {
            'recovery_prediction': 'recovery',
            'recovery': 'recovery',
            'stroke_classification': 'classification',
            'classification': 'classification'
        }
        
        # Map strategy names
        strategy_mapping = {
            'zero_shot': 'zero_shot',
            'chain_of_thought': 'cot',
            'cot': 'cot',
            'few_shot': 'few_shot'
        }
        
        task_key = task_mapping.get(task, task)
        strategy_key = strategy_mapping.get(strategy, strategy)
        
        file_name = f"{task_key}_{strategy_key}.txt"
        file_path = self.prompts_dir / file_name
        
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def format_prompt(self, task: str, strategy: str, patient_data: Dict[str, Any], 
                     n_shots: int = 3) -> str:
        """
        Format a prompt with patient data and examples
        
        Args:
            task: Task name
            strategy: Prompting strategy
            patient_data: Patient clinical data
            n_shots: Number of examples for few-shot (ignored for other strategies)
            
        Returns:
            Formatted prompt string
        """
        template = self.load_prompt_template(task, strategy)
        
        # Format patient data
        patient_str = self._format_patient_data(patient_data)
        
        if strategy == 'few_shot':
            # Get examples for few-shot prompting
            examples_str = self._format_examples(task, n_shots)
            return template.format(examples=examples_str, patient_data=patient_str)
        else:
            return template.format(patient_data=patient_str)
    
    def _format_patient_data(self, patient_data: Dict[str, Any]) -> str:
        """Format patient data into readable string"""
        formatted_parts = []
        
        for key, value in patient_data.items():
            # Format key names for readability
            readable_key = key.replace('_', ' ').title()
            
            # Handle different value types
            if isinstance(value, (int, float)):
                if key in ['age', 'stroke_episodes']:
                    formatted_parts.append(f"{readable_key}: {int(value)}")
                else:
                    formatted_parts.append(f"{readable_key}: {value}")
            elif isinstance(value, bool):
                formatted_parts.append(f"{readable_key}: {'Yes' if value else 'No'}")
            else:
                formatted_parts.append(f"{readable_key}: {value}")
        
        return ", ".join(formatted_parts)
    
    def _format_examples(self, task: str, n_shots: int) -> str:
        """Format few-shot examples"""
        task_key = 'recovery_prediction' if 'recovery' in task else 'stroke_classification'
        
        if task_key not in self.examples:
            return ""
        
        examples = self.examples[task_key][:n_shots]
        formatted_examples = []
        
        for i, example in enumerate(examples, 1):
            if task_key == 'recovery_prediction':
                patient_str = self._format_patient_data(example['patient_data'])
                formatted_examples.append(
                    f"Example {i}:\n"
                    f"Patient: {patient_str}\n"
                    f"Reasoning: {example['clinical_reasoning']}\n"
                    f"Recovery Score: {example['recovery_score']}\n"
                )
            else:  # stroke_classification
                patient_str = self._format_patient_data(example['patient_data'])
                formatted_examples.append(
                    f"Example {i}:\n"
                    f"Patient: {patient_str}\n"
                    f"Reasoning: {example['clinical_reasoning']}\n"
                    f"Level 1: {example['level1_classification']}\n"
                    f"Level 2: {example['level2_subtype']}\n"
                )
        
        return "\n".join(formatted_examples)
    
    def get_available_prompts(self) -> Dict[str, List[str]]:
        """Get list of available prompt templates"""
        available = {}
        
        for task in ['recovery', 'classification']:
            available[task] = []
            for strategy in ['zero_shot', 'cot', 'few_shot']:
                file_name = f"{task}_{strategy}.txt"
                file_path = self.prompts_dir / file_name
                if file_path.exists():
                    available[task].append(strategy)
        
        return available
    
    def get_example_categories(self, task: str) -> List[str]:
        """Get available example categories for a task"""
        task_key = 'recovery_prediction' if 'recovery' in task else 'stroke_classification'
        
        if task_key not in self.examples:
            return []
        
        if task_key == 'recovery_prediction':
            return list(set(ex.get('outcome_category', 'unknown') for ex in self.examples[task_key]))
        else:
            return list(set(ex.get('confidence', 'unknown') for ex in self.examples[task_key]))
    
    def get_examples_by_category(self, task: str, category: str, 
                                n_examples: int = 5) -> List[Dict]:
        """Get examples filtered by category"""
        task_key = 'recovery_prediction' if 'recovery' in task else 'stroke_classification'
        
        if task_key not in self.examples:
            return []
        
        if task_key == 'recovery_prediction':
            filtered = [ex for ex in self.examples[task_key] 
                       if ex.get('outcome_category') == category]
        else:
            filtered = [ex for ex in self.examples[task_key] 
                       if ex.get('confidence') == category]
        
        return filtered[:n_examples]
    
    def validate_prompt_output(self, task: str, output: str) -> bool:
        """Validate LLM output format"""
        if not output or len(output.strip()) == 0:
            return False
        
        if task in ['recovery', 'recovery_prediction']:
            # Check for numerical score
            import re
            numbers = re.findall(r'\b\d{1,3}\b', output)
            valid_scores = [int(n) for n in numbers if 0 <= int(n) <= 100]
            return len(valid_scores) > 0
        
        elif task in ['classification', 'stroke_classification']:
            # Check for classification keywords
            output_lower = output.lower()
            level1_keywords = ['ischemic', 'hemorrhagic', 'undefined']
            return any(keyword in output_lower for keyword in level1_keywords)
        
        return True
    
    def extract_prediction(self, task: str, output: str) -> Any:
        """Extract prediction from LLM output"""
        if task in ['recovery', 'recovery_prediction']:
            import re
            numbers = re.findall(r'\b\d{1,3}\b', output)
            scores = [int(n) for n in numbers if 0 <= int(n) <= 100]
            return scores[-1] if scores else 50  # Default to 50 if no valid score
        
        elif task in ['classification', 'stroke_classification']:
            output_lower = output.lower()
            
            # Level 1 classification
            if 'ischemic' in output_lower:
                level1 = 'Ischemic'
            elif 'hemorrhagic' in output_lower:
                level1 = 'Hemorrhagic'
            else:
                level1 = 'Undefined'
            
            # Level 2 classification
            level2_mapping = {
                'infarction': 'Cerebral Infarction',
                'hemorrhage': 'Cerebral Hemorrhage', 
                'thrombosis': 'Cerebral Thrombosis',
                'embolism': 'Cerebral Embolism'
            }
            
            level2 = 'Other'
            for keyword, subtype in level2_mapping.items():
                if keyword in output_lower:
                    level2 = subtype
                    break
            
            return {'level1': level1, 'level2': level2}
        
        return None


def load_prompt_templates(prompts_dir: str = "prompts") -> PromptLoader:
    """Convenience function to create and return a PromptLoader instance"""
    return PromptLoader(prompts_dir)


# Example usage
if __name__ == "__main__":
    # Initialize prompt loader
    loader = PromptLoader()
    
    # Example patient data
    patient_data = {
        'age': 68,
        'gender': 'Male',
        'd_dimer': 0.8,
        'hemoglobin': 140,
        'crp': 3.2,
        'hypertension': True,
        'diabetes': False
    }
    
    # Load and format prompts
    try:
        # Zero-shot recovery prediction
        recovery_prompt = loader.format_prompt(
            'recovery_prediction', 'zero_shot', patient_data
        )
        print("Recovery Prediction (Zero-shot):")
        print(recovery_prompt[:200] + "...")
        print()
        
        # Chain-of-thought classification
        classification_prompt = loader.format_prompt(
            'stroke_classification', 'chain_of_thought', patient_data
        )
        print("Classification (Chain-of-thought):")
        print(classification_prompt[:200] + "...")
        print()
        
        # Few-shot recovery prediction
        few_shot_prompt = loader.format_prompt(
            'recovery_prediction', 'few_shot', patient_data, n_shots=3
        )
        print("Recovery Prediction (Few-shot):")
        print(few_shot_prompt[:200] + "...")
        print()
        
        # Show available prompts
        available = loader.get_available_prompts()
        print("Available prompts:", available)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all prompt files are in the prompts/ directory")
