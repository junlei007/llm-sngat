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

### Data Splits

The dataset contains:
- **Training set**: 97,467 problems
- **Development set**: 254 problems  
- **Test set**: 254 problems

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

# Custom configuration
python run_experiment.py \
  --models GPT-4o DeepSeek-R1 \
  --n-students 200 \
  --form-size 60 \
  --common-sizes 5 10 15 20 25 \
  --replications 3 \
  --dataset-path data/aqua_rat
```

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
- Check the [documentation](docs/)
- Review the [demo notebook](demo.ipynb)

---

**Note**: This implementation provides simplified standard error approximations for demonstration. For publication-quality research, use the CIPE software integration for precise Delta method calculations.