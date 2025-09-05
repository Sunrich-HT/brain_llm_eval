# Stroke LLM Evaluation Framework

## Evaluating Large Language Models for Recovery Prediction and Diagnostic Classification in Stroke Neurology


## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- API keys for LLM services (optional, for LLM evaluation)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/stroke-llm-evaluation.git
cd stroke-llm-evaluation
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables** (create `.env` file)
```bash
# Optional: Add your API keys for LLM evaluation
echo "OPENAI_API_KEY=your_openai_key_here" > .env
echo "ANTHROPIC_API_KEY=your_anthropic_key_here" >> .env
echo "GOOGLE_API_KEY=your_google_key_here" >> .env
```

## ⚡ Quick Start

### Option 1: Full Evaluation (requires API keys)
```bash
# Place your data file in the root directory
# 浦沿卒中随访患者结果.xls

# Run complete evaluation
python run_evaluation.py
```

### Option 2: Baseline Models Only
```python
from stroke_llm_evaluation import DataPreprocessor, BaselineEvaluator

# Load and preprocess data
processor = DataPreprocessor('浦沿卒中随访患者结果.xls')
df = processor.load_data()
X, y_reg, y_cls = processor.preprocess_data(df)

# Evaluate baseline models
evaluator = BaselineEvaluator()
results = evaluator.evaluate_baselines(X, y_reg, y_cls)
print(results)
```

### Option 3: Custom LLM Evaluation
```python
from stroke_llm_evaluation import LLMEvaluator, ModelConfig

# Configure your models
model_configs = {
    'gpt-4o': ModelConfig('gpt-4o', 'your-api-key'),
    'claude-sonnet-4': ModelConfig('claude-sonnet-4', 'your-api-key')
}

# Run evaluation
evaluator = LLMEvaluator(model_configs)
result = evaluator.evaluate_model('gpt-4o', df_sample, 'recovery', 'zero_shot')
```

## 📁 Project Structure

```
stroke-llm-evaluation/
├── 📄 README.md                          # This file
├── 📄 LICENSE                            # License information
├── 📄 requirements.txt                   # Python dependencies
├── 📄 config.py                          # Configuration settings
├── 📄 stroke_llm_evaluation.py          # Main evaluation framework
├── 📄 run_evaluation.py                 # Simple runner script
├── 📊 浦沿卒中随访患者结果.xls           # Raw clinical data
├── 📁 results/                           # Evaluation results
│   ├── llm_evaluation_results.csv       # LLM performance data
│   └── baseline_evaluation_results.csv  # Baseline performance data
├── 📁 plots/                            # Generated visualizations
│   ├── model_comparison.png            # Model performance comparison
│   └── cross_task_correlation.png      # Cross-task analysis
├── 📁 logs/                             # Execution logs
├── 📁 data/                             # Processed data
└── 📁 prompts/                          # Prompt templates (optional)
```

## ⚙️ Configuration

### API Keys Setup

**Method 1: Environment Variables**
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
```

**Method 2: Configuration File**
Edit `config.py` to add your API keys:
```python
class APIConfig:
    OPENAI_API_KEY = "your-openai-key-here"
    ANTHROPIC_API_KEY = "your-anthropic-key-here"
    GOOGLE_API_KEY = "your-google-key-here"
```

### Evaluation Settings
Modify `EVALUATION_CONFIG` in `config.py`:
```python
EVALUATION_CONFIG = {
    'n_runs': 3,                    # Runs per evaluation
    'sample_size': 152,             # Full dataset size
    'test_sample_size': 50,         # LLM evaluation subset
    'random_state': 42,
    'rate_limit_delay': 0.1,        # Seconds between API calls
    'max_retries': 3
}
```

## 📖 Usage

### Command Line Interface
```bash
# Run full evaluation
python run_evaluation.py

# Run with custom sample size
python -c "
import sys
sys.path.append('.')
from config import EVALUATION_CONFIG
EVALUATION_CONFIG['test_sample_size'] = 20
exec(open('run_evaluation.py').read())
"
```

### Programmatic Usage

**1. Data Processing**
```python
from stroke_llm_evaluation import DataPreprocessor

processor = DataPreprocessor('data.xls')
df = processor.load_data()
X, y_regression, y_classification = processor.preprocess_data(df)
```

**2. Baseline Evaluation**
```python
from stroke_llm_evaluation import BaselineEvaluator

evaluator = BaselineEvaluator()
results = evaluator.evaluate_baselines(X, y_regression, y_classification)
```

**3. LLM Evaluation**
```python
from stroke_llm_evaluation import LLMEvaluator, ModelConfig

configs = {'gpt-4o': ModelConfig('gpt-4o', 'api-key')}
evaluator = LLMEvaluator(configs)

# Single evaluation
result = evaluator.evaluate_model('gpt-4o', df, 'recovery', 'zero_shot')

# Batch evaluation
for task in ['recovery', 'classification']:
    for strategy in ['zero_shot', 'cot', 'few_shot_3']:
        result = evaluator.evaluate_model('gpt-4o', df, task, strategy)
```

**4. Statistical Analysis**
```python
from stroke_llm_evaluation import StatisticalAnalyzer

analyzer = StatisticalAnalyzer()
stats = analyzer.compare_models(llm_results, baseline_results)
print(f"Recovery task p-value: {stats['recovery_comparison']['p_value']}")
```

**5. Visualization**
```python
from stroke_llm_evaluation import Visualizer

viz = Visualizer()
viz.plot_model_comparison(llm_results, baseline_results)
viz.plot_cross_task_correlation(llm_results)
```

## 🎯 Evaluation Tasks

### Task 1: Functional Recovery Prediction
- **Input**: 27 clinical features (demographics, lab values, vital signs, clinical conditions)
- **Output**: Recovery score (0-100 continuous scale)
- **Metric**: Normalized Mean Absolute Error Score (NMAE)
- **Clinical Context**: Post-acute care rehabilitation planning

### Task 2: Hierarchical Stroke Classification
- **Input**: Same 27 clinical features
- **Output**: Two-level classification
  - Level 1: Ischemic/Hemorrhagic/Undefined
  - Level 2: Specific stroke subtype
- **Metrics**: First-level accuracy, Overall accuracy, Macro-F1, Balanced Accuracy
- **Clinical Context**: Diagnostic workup and treatment planning

### Clinical Features (27 total)
- **Demographics**: Age, gender, living situation, education
- **Laboratory**: D-dimer, hemoglobin, CRP, albumin, uric acid, glucose
- **Vital Signs**: BMI, blood pressure (systolic/diastolic)
- **Clinical**: Hypertension, diabetes, stroke episodes, exercise habits

## 🤖 Models Supported

### Proprietary Models
- **OpenAI**: GPT-4o, OpenAI o3
- **Anthropic**: Claude Opus 4, Claude Sonnet 4
- **Google**: Gemini 2.5 Pro, Gemini 2.5 Flash

### Emerging Models
- **xAI**: Grok-3-beta

### Open Source Models
- **Meta**: LLaMA-4-Maverick, LLaMA-3.3-70B
- **Alibaba**: Qwen-2.5-Max, Qwen-3-235B
- **DeepSeek**: DeepSeek-V3, DeepSeek-R1

### Baseline Models
- **Random Forest**: Ensemble learning baseline
- **XGBoost**: Gradient boosting baseline
- **TabNet**: Deep learning baseline for tabular data

## 📊 Results

### Key Findings
- **Best Recovery Prediction**: Grok-3-beta (NMAE: 0.928) with Chain-of-Thought
- **Best Classification**: OpenAI o3 (85.2% first-level, 77.5% overall)
- **CoT Improvement**: Consistent 2-3 percentage point gains across models
- **LLM vs Baseline**: 5-10 percentage point advantage (p < 0.001)

### Output Files
- `results/llm_evaluation_results.csv`: Detailed LLM performance
- `results/baseline_evaluation_results.csv`: Traditional ML results
- `plots/model_comparison.png`: Performance visualization
- `plots/cross_task_correlation.png`: Cross-task analysis

### Sample Results Format
```csv
Model,Task,Strategy,Performance,MAE,Correlation
gpt-4o,recovery,zero_shot,0.8928,10.72,0.8712
gpt-4o,recovery,cot,0.9065,9.35,0.8893
gpt-4o,classification,zero_shot,0.7974,,,
gpt-4o,classification,cot,0.8224,,,
```

## 💰 Cost Management

### API Cost Optimization
- **Subset Sampling**: Configurable sample sizes for LLM evaluation
- **Rate Limiting**: Automatic delays between API calls
- **Error Handling**: Retry logic to minimize failed requests
- **Batch Processing**: Efficient evaluation scheduling

### Estimated Costs (per 152 samples)
- **GPT-4o**: ~$15-25 per complete evaluation
- **Claude**: ~$10-20 per complete evaluation
- **Gemini**: ~$5-15 per complete evaluation

*Note: Costs vary by model and prompt length. Use test_sample_size for budget control.*

## 🧪 Testing

### Run Tests
```bash
# Test data processing
python -c "
from stroke_llm_evaluation import DataPreprocessor
processor = DataPreprocessor('test_data.xls')
df = processor.load_data()
print(f'Data shape: {df.shape}')
"

# Test baseline models
python -c "
from stroke_llm_evaluation import BaselineEvaluator, DataPreprocessor
processor = DataPreprocessor('test_data.xls')
df = processor.load_data()
X, y_reg, y_cls = processor.preprocess_data(df)
evaluator = BaselineEvaluator()
results = evaluator.evaluate_baselines(X, y_reg, y_cls)
print('Baseline test passed')
"
```

### Validation Checklist
- [ ] Data loads without errors
- [ ] All 27 features present
- [ ] Baseline models run successfully
- [ ] API calls work (if keys provided)
- [ ] Results save to CSV files
- [ ] Plots generate correctly



## 📄 License

This project is licensed under the MIT License.

## 🏥 Clinical Disclaimer

This software is for research purposes only and is not intended for clinical diagnosis or treatment decisions. Always consult qualified healthcare professionals for medical advice.

## 🙏 Acknowledgments

- OpenAI for GPT-4o and o3 API access
- Anthropic for Claude API access
- Google for Gemini API access
- The stroke neurology research community
- All reviewers and contributors
