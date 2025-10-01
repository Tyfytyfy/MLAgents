# MLAgents

Machine Learning pipeline using AI agents for automated data analysis.

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

Use any analysis file as a template (e.g., `examples/breast_cancer/breast_cancer_analysis.py`). 

Modify these 3 lines:
1. `data = pd.read_csv('data.csv')` - change to your CSV file path
2. In `initial_state` dict: `"data_file_path": "data.csv"` - change to your file path  
3. In `initial_state` dict: `"user_objective"` - change to your analysis goal

Example:
```python
"user_objective": "I want to classify breast cancers with accuracy of 80%"
```

## Example Files
- `examples/breast_cancer/breast_cancer_analysis.py`
- `examples/heart_failure/heart_failure_analysis.py` 
- `examples/wine_quality/red_wine_analysis.py`