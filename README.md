# LLM-SNGAT: LLM-Simulated Nonequivalent Groups with Anchor Test

This repository contains a Python implementation of the **LLM-Simulated Nonequivalent Groups with Anchor Test (LLM-SNGAT)** methodology for test equating, as described in the research paper on using Large Language Models to simulate test-taker response patterns.

## Overview

The LLM-SNGAT method addresses the challenge of test equating when traditional anchor items are not available. It uses LLMs to simulate authentic test-taker responses, transforming non-anchor equating problems into common-item nonequivalent groups designs.

### Key Features

- **Two-stage simulation process**: 
  1. Simulate responses to original test forms without anchors
  2. Implement LLM-SNGAT with generated common items
- **Multiple LLM support**: Compatible with GPT-4o, O1-preview, DeepSeek-R1, and other models
- **Psychometric equating methods**: Tucker linear and Levine observed score equating
- **Comprehensive analysis**: Standard error calculations by raw score and visualization tools
- **Configurable anchor sizes**: Support for different common item configurations (18%, 33%, 46%, 57%)
- **CIPE integration**: Export data for precise standard error calculation using CIPE software

## Dataset: AQuA-RAT (Algebra Question Answering with Rationales)

This implementation uses the [AQuA-RAT dataset](https://github.com/google-deepmind/AQuA) from Google DeepMind.

### About AQuA-RAT

The AQuA-RAT dataset consists of approximately 100,000 algebraic word problems with natural language rationales. Each problem is a JSON object consisting of four parts:
- **question**: A natural language definition of the problem to solve
- **options**: 5 possible options (A, B, C, D and E), among which one is correct  
- **rationale**: A natural language description of the solution to the problem
- **correct**: The correct option

### Dataset Structure

```json
{
  "question": "A grocery sells a bag of ice for $1.25, and makes 20% profit. If it sells 500 bags of ice, how much total profit does it make?",
  "options": ["A)125", "B)150", "C)225", "D)250", "E)275"],
  "rationale": "Profit per bag = 1.25 * 0.20 = 0.25\nTotal profit = 500 * 0.25 = 125\nAnswer is A.",
  "correct": "A"
}
```

### Test Form Creation Strategy

Unlike traditional approaches that use the entire dataset, this implementation follows a more realistic testing scenario by sampling a subset of problems to create test forms:

- **Form X**: 50 problems sampled from AQuA dataset
- **Form Y**: 50 problems sampled from AQuA dataset  
- **No overlap**: Ensures statistical independence between forms
- **Sampling methods**: 
  - Random sampling (default)
  - Stratified sampling (optional, balances difficulty)

### Sample Data from Research Paper

The repository includes the actual mathematical problems used in the research paper, stored in `data/sample_items.txt`. This file contains 100 carefully selected AQuA problems:

- **Questions 1-50**: Form X problems
- **Questions 51-100**: Form Y problems

Each problem follows the standardized format:
```json
{"idx": "Q1", "question": "The vertex of a parallelogram are (1, 0), (3, 0), (1, 1) and (3, 1) respectively. If line L passes through the origin and divided the parallelogram into two identical quadrilaterals, what is the slope of line L?", "options": ["A). 1/2", "B)2", "C)1/4", "D)3", "E)3/4"]}
```

This sample dataset can be used to:
- Replicate the exact experimental conditions from the research paper
- Test the LLM-SNGAT methodology with known problem sets
- Validate equating results against published findings
- Demonstrate the system without requiring the full AQuA dataset download

### Response Format for LLM Simulation

When using real LLMs for response simulation, the system uses a standardized JSON response format:

```json
{
  "Q1": "A",
  "Q2": "B", 
  "Q3": "D",
  ...
  "Q50": "C"
}
```

This format ensures:
- Clean parsing of LLM responses
- Consistent data structure across different models
- Easy validation and error handling
- Compatibility with the equating analysis pipeline

### Using the Research Paper Sample Data

To use the exact problems from the research paper:

```python
# Load the research sample data
dataset_loader = AQuaDatasetLoader('data/sample_items.txt')

# Or load through the project structure
form_x, form_y = load_sample_forms_from_paper()
```

### Example Usage with Different Data Sources

```python
# Using full AQuA dataset with random sampling
dataset_loader = AQuaDatasetLoader('data/AQuA/train.json')
form_x, form_y = dataset_loader.create_test_forms(form_size=50)

# Using research paper's exact sample
dataset_loader = AQuaDatasetLoader('data/sample_items.txt')
form_x, form_y = dataset_loader.create_test_forms(form_size=50)

# Using stratified sampling (balanced difficulty)
form_x, form_y = dataset_loader.create_stratified_test_forms(form_size=50)

# Get sampling statistics
stats = dataset_loader.get_sampling_statistics(form_x, form_y)
print(f"Sampling rate: {stats['sampling_rate']:.2f}%")
```

### Citation

```bibtex
@inproceedings{ling2017program,
  title={Program induction by rationale generation: Learning to solve and explain algebraic word problems},
  author={Ling, Wang and Yogatama, Dani and Dyer, Chris and Blunsom, Phil},
  booktitle={Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={158--167},
  year={2017}
}
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-sngat.git
cd llm-sngat
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the AQuA dataset:
```bash
# Option 1: Clone the original repository
git clone https://github.com/google-deepmind/AQuA.git data/AQuA

# Option 2: Use Hugging Face datasets
pip install datasets
python -c "from datasets import load_dataset; dataset = load_dataset('deepmind/aqua_rat', 'raw'); dataset.save_to_disk('data/aqua_rat')"
```

4. (Optional) Set up OpenAI API key for actual LLM simulation:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from llm_sngat import *

# Initialize components with AQuA dataset
dataset_loader = AQuaDatasetLoader('data/AQuA/train.json')  # or your dataset path
simulator = LLMResponseSimulator()
processor = LLMSNGATProcessor(dataset_loader, simulator)

# Run LLM-SNGAT analysis
form_x, form_y, students_x, students_y = processor.stage_one_simulation(
    n_students=150, form_size=50
)

# Analyze with different anchor sizes
for n_common in [5, 10, 15, 20]:
    new_form_x, new_form_y, common_x, common_y = processor.stage_two_simulation(
        form_x, form_y, students_x, students_y, n_common
    )
    
    # Calculate scores and perform equating
    scores_x = processor.calculate_scores(students_x, new_form_x, 'responses_x')
    scores_y = processor.calculate_scores(students_y, new_form_y, 'responses_y')
    
    equating_results = processor.tucker_levine_equating(scores_x, scores_y, ...)
```

### Command Line Usage

```bash
# Run demonstration with sample data
python llm_sngat.py

# Run comprehensive experiment with real AQuA data
python run_experiment.py \
  --models GPT-4o O1-preview DeepSeek-R1 \
  --dataset-path data/AQuA/train.json \
  --use-real-llm

# Custom configuration with stratified sampling
python run_experiment.py \
  --models GPT-4o DeepSeek-R1 \
  --n-students 200 \
  --form-size 60 \
  --common-sizes 5 10 15 20 25 \
  --replications 3 \
  --dataset-path data/aqua_rat \
  --use-stratified

# Quick development test with limited dataset
python run_experiment.py \
  --max-dataset-size 500 \
  --form-size 25 \
  --models GPT-4o
```

### Research Implementation Note

This implementation is designed to replicate the methodology described in academic research. The sampling approach (50 problems per test form) reflects realistic testing scenarios while maintaining the statistical rigor required for psychometric equating analysis.

**Key advantages of the sampling approach:**
- **Computational efficiency**: Faster processing compared to using entire dataset
- **Realistic scale**: Matches typical standardized test lengths (50-100 items)
- **Experimental control**: Reproducible results through seed-based sampling
- **Flexibility**: Easy to adjust form size based on research requirements

### Jupyter Notebook Demo

```bash
jupyter notebook demo.ipynb
```

## Standard Error Analysis

The implementation provides standard errors calculated at each raw score level, similar to CIPE software output:

```
 STANDARD ERRORS OF EQUATING
============================================================
 X-SCORE       TUCKER       LEVINE
------------------------------------------------------------
       0       1.8366       3.3135
       1       1.7779       3.2072
       2       1.7192       3.1009
       ...
      50       1.1693       2.0802
------------------------------------------------------------
 AVERAGE       1.2345       2.1234
```

### LLM Response Collection

For real LLM implementation, the system collects responses using a standardized prompt that instructs the model to:
1. Role-play as a student with specified mathematical ability
2. Consider demographic characteristics (gender, location)
3. Respond in JSON format for clean parsing

**Example LLM prompt template:**
```
请扮演一名来自中国北京的学生，姓名是张明。该学生的性别是男。该学生的数学能力是75.0，最高能力为100，能力越高表示回答正确的概率越高。

请不要输出思考过程，只输出答案并且按json格式输出答案，示例如下：
{"Q1":"A", "Q2":"B", "Q3":"D"}

[问题列表]
```

This approach ensures:
- Consistent response format across different LLMs
- Clean data extraction and processing
- Reproducible experimental conditions
- Scalable data collection for large-scale studies

### Integration with CIPE Software

For precise standard error calculations, the system can export data in CIPE-compatible format:

```python
# Export data for CIPE analysis
analyzer.export_for_cipe(
    students_x, students_y, new_form_x, new_form_y,
    common_indices_x, common_indices_y, 'cipe_export'
)
```

Download CIPE software from: [University of Iowa CASMA](https://www.education.uiowa.edu/centers/casma/computer-programs)

## Project Structure

```
llm-sngat/
├── llm_sngat.py              # Main implementation
├── config.py                 # Configuration management  
├── utils.py                  # Utility functions
├── run_experiment.py         # Experiment runner
├── demo.ipynb               # Jupyter demonstration
├── data/                    # Dataset directory
│   ├── AQuA/               # Original AQuA dataset
│   └── aqua_rat/           # Hugging Face format
├── results/                # Output results
├── figures/               # Generated plots
└── cipe_export/          # CIPE-compatible data
```

## Output Files

The system generates comprehensive outputs:

- **Excel Reports**: `results/llm_sngat_results_*.xlsx`
- **Visualization**: `figures/model_comparison_*.png`, `figures/standard_errors_by_score.png`
- **CIPE Data**: `cipe_export/group_*_responses.dat`
- **Standard Error Tables**: `results/standard_errors_detailed.txt`

## Dataset Loading Options

### Option 1: Original AQuA Repository
```python
dataset_loader = AQuaDatasetLoader('data/AQuA/train.json')
```

### Option 2: Hugging Face Datasets
```python
from datasets import load_dataset
dataset = load_dataset('deepmind/aqua_rat', 'raw')
dataset_loader = AQuaDatasetLoader(dataset['train'])
```

### Option 3: Sample Data (No Download Required)
```python
dataset_loader = AQuaDatasetLoader()  # Uses generated sample problems
```

## Model Support

- **GPT-4o**: OpenAI's latest model
- **O1-preview**: OpenAI's reasoning model  
- **DeepSeek-R1**: DeepSeek's reasoning model
- **Claude**: Anthropic's models
- **Custom Models**: Easily extensible for new LLMs

## Research Applications

This implementation is designed for:
- Educational test equating research
- LLM evaluation in mathematical reasoning
- Psychometric method validation
- Test development without traditional anchor items

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The AQuA dataset is licensed under Apache License 2.0 - see the [original repository](https://github.com/google-deepmind/AQuA) for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{llm_sngat,
  title={LLM-SNGAT: LLM-Simulated Nonequivalent Groups with Anchor Test},
  author={Research Team},
  year={2025},
  url={https://github.com/yourusername/llm-sngat}
}
```

And the original AQuA dataset:

```bibtex
@inproceedings{ling2017program,
  title={Program induction by rationale generation: Learning to solve and explain algebraic word problems},
  author={Ling, Wang and Yogatama, Dani and Dyer, Chris and Blunsom, Phil},
  booktitle={Proceedings of ACL},
  year={2017}
}
```

## Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

For questions and support:
- Create an issue in this repository
- Review the [demo notebook](demo.ipynb)

---

**Note**: This implementation provides simplified standard error approximations for demonstration. For publication-quality research, use the CIPE software integration for precise Delta method calculations.