import re
from datetime import timedelta

from states.data_state import CodeGenerationMessage, MLState
from utils.extract_text import extract_hyperparameters_from_code, extract_algorithm_from_code, \
    extract_model_type_from_code


def create_no_change_focus_generator_prompt(user_objective: str, data_analysis: dict, target_analysis: dict,
                                            feature_engineering: dict):
    return f"""
   You are an expert Python programmer specializing in machine-learning pipelines. Your task is to create an ML pipeline.

   User objective: {user_objective}
   Target Variable: {target_analysis.get('target_variable', '')}
   Data profile: {data_analysis.get('profile', '')}
   Available columns: {data_analysis.get('columns', [])}
   FEATURE ENGINEERING REQUIREMENTS:
   - Feature Plan: {feature_engineering.get('plan', '')}
   - Preprocessing Steps: {feature_engineering.get('preprocessing_steps', [])}
   - Scaling Method: {feature_engineering.get('scaling_method', 'StandardScaler')}
   - Encoding Method: {feature_engineering.get('encoding_method', '')}
   - Features to Drop: {feature_engineering.get('features_to_drop', [])}

   MANDATORY INSTRUCTIONS:
   1. Implement the Plan: You must implement the feature engineering plan exactly as specified.
   2. Use Specified Methods: Use {feature_engineering.get('scaling_method', 'StandardScaler')} for scaling, implement all preprocessing steps {feature_engineering.get('preprocessing_steps', [])}, and drop the specified features {feature_engineering.get('features_to_drop', [])}.
   3. Generate a Complete Script: The code must be a single, runnable Python script that loads 'data.csv' and prints the final model accuracy.

   MANDATORY CODE STRUCTURE:
   Your generated_code MUST include these marker sections:

   # === IMPORTS_START ===
   all import statements
   # === IMPORTS_END ===

   # === DATA_LOADING_START ===
   data loading and initial preparation
   # === DATA_LOADING_END ===

   # === FEATURE_ENGINEERING_START ===
   feature engineering and preprocessing steps
   # === FEATURE_ENGINEERING_END ===

   # === DATA_SPLITTING_START ===
   train-test split
   # === DATA_SPLITTING_END ===

   # === SCALING_START ===
   scaling/normalization code
   # === SCALING_END ===

    # === ALGORITHM_SELECTION_START ===
    algorithm import statement
    # === ALGORITHM_SELECTION_END ===
    
    # === HYPERPARAMETERS_START ===
    model = AlgorithmName(parameters)
    # === HYPERPARAMETERS_END ===

   # === MODEL_TRAINING_START ===
   model training code
   # === MODEL_TRAINING_END ===

   # === PREDICTION_START ===
   prediction and evaluation code
   # === PREDICTION_END ===

   EXAMPLES:
   # === IMPORTS_START ===
   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.preprocessing import StandardScaler
   # === IMPORTS_END ===

    # === ALGORITHM_SELECTION_START ===
    from sklearn.ensemble import RandomForestClassifier
    # === ALGORITHM_SELECTION_END ===
    
    # === HYPERPARAMETERS_START ===
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    # === HYPERPARAMETERS_END ===

   These markers are MANDATORY and enable precise code modifications without regenerating entire scripts.

   Return ONLY a valid JSON object in the following format:
   {{
       "generated_code": "A complete Python script that implements the feature engineering requirements with ALL marker sections.",
       "pipeline_code": "The part of the script that defines the scikit-learn preprocessing pipeline.",
       "model_code": "The part of the script that trains the model and makes predictions."
   }}
   """
def create_change_focus_generator_prompt(user_objective: str, data_analysis: dict, target_analysis: dict, feature_engineering: dict,
                                         change_focus: str, previous_approach, hyperparams: dict = None):
    if change_focus == "hyperparameter_tuning":
        if hyperparams is None:
            hyperparams = extract_hyperparameters_from_code(previous_approach)
        param_options = get_parameter_options(data_analysis)
        return f"""
        User objective: {user_objective}
        Target Variable: {target_analysis.get('target_variable', '')}
        Data profile: {data_analysis.get('profile', '')}
        Available columns: {data_analysis.get('columns', [])}
        FEATURE ENGINEERING REQUIREMENTS:
        - Feature Plan: {feature_engineering.get('plan', '')}
        - Preprocessing Steps: {feature_engineering.get('preprocessing_steps', [])}
        - Scaling Method: {feature_engineering.get('scaling_method', 'StandardScaler')}
        - Encoding Method: {feature_engineering.get('encoding_method', '')}
        - Features to Drop: {feature_engineering.get('features_to_drop', [])}
    
        Current hyperparameters: {hyperparams}

        AVAILABLE PARAMETER OPTIONS (choose from these only):
        - n_estimators: {param_options['n_estimators']} (current: {hyperparams.get('n_estimators', 100)})
        - max_depth: {param_options['max_depth']} (current: {hyperparams.get('max_depth', 'None')})
    
        CHANGE SPECIFIC PARAMETERS:
        - Choose n_estimators from available options that's different from current
        - Choose max_depth from available options that's different from current
    

        Generate COMPLETE WORKING code that:
        1. Loads 'data.csv' using pd.read_csv  
        2. Implements the EXACT preprocessing from previous solution
        3. Only changes hyperparameters: n_estimators=200, max_depth=10
        4. Must be runnable without modifications
        Return ONLY valid JSON:
       {{
            "generated_code": "Complete script with same preprocessing, DIFFERENT algorithm",
           "pipeline_code": "Identical preprocessing pipeline from previous solution", 
           "model_code": "NEW algorithm with default parameters",
       }}
        """
    elif change_focus == "model_algorithm":
        return f"""
            User objective: {user_objective}
            Target Variable: {target_analysis.get('target_variable', '')}
            Data profile: {data_analysis.get('profile', '')}
            Available columns: {data_analysis.get('columns', [])}
            FEATURE ENGINEERING REQUIREMENTS:
            - Feature Plan: {feature_engineering.get('plan', '')}
            - Preprocessing Steps: {feature_engineering.get('preprocessing_steps', [])}
            - Scaling Method: {feature_engineering.get('scaling_method', 'StandardScaler')}
            - Encoding Method: {feature_engineering.get('encoding_method', '')}
            - Features to Drop: {feature_engineering.get('features_to_drop', [])}

               Current algorithm: {previous_approach}

               CHANGE ONLY: Machine learning algorithm
               - From current: {previous_approach}
               - To different algorithm: RandomForest -> XGBoost, LogisticRegression -> SVM, etc.
               - Use default hyperparameters for new algorithm
               
               Generate code with IDENTICAL preprocessing but DIFFERENT algorithm.

               Return ONLY valid JSON:
               {{
                    "generated_code": "Complete script with same preprocessing, DIFFERENT algorithm",
                   "pipeline_code": "Identical preprocessing pipeline from previous solution", 
                   "model_code": "NEW algorithm with default parameters",
               }}
               """
    else:
        return create_no_change_focus_generator_prompt(user_objective, data_analysis, target_analysis, feature_engineering)

def create_code_generation_message(config: dict, context_text: str) -> CodeGenerationMessage:
    return CodeGenerationMessage(
        role="assistant",
        content=context_text,
        type="code_generation",
        generated_code=config["generated_code"],
        pipeline_code=config["pipeline_code"],
        model_code=config["model_code"],
        hyperparameters=extract_hyperparameters_from_code(config['generated_code']),
    )

def create_code_generation_return_state(state: MLState, code_message: CodeGenerationMessage, response, duration) -> MLState:
    total_tokens = 0
    if response is not None:
        total_tokens = response.usage_metadata.get('total_tokens', 0)
    return {
        "data_file_path": state.get("data_file_path", ""),
        "data_summary": state.get("data_summary", ""),
        "user_objective": state.get("user_objective", ""),
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 3),
        "improvement_history": state.get("improvement_history", []),
        "messages": state.get('messages', []) + [code_message],
        "data_analysis": state.get("data_analysis", {}),
        "target_analysis": state.get("target_analysis", {}),
        "feature_engineering": state.get("feature_engineering", {}),
        "solution_history": state.get("solution_history", []),
        "current_best_solution": state.get("current_best_solution"),
        "change_focus": state.get("change_focus"),
        "execution_time": state.get("execution_time", timedelta(0)) + timedelta(seconds=duration),
        "total_tokens": state.get("total_tokens", 0) + total_tokens
    }


def get_parameter_options(data_analysis: dict):
    data_len = data_analysis["data_length"]

    if data_len <= 500:
        size = 'small'
    elif data_len <= 10000:
        size = 'large'
    else:
        size = 'medium'

    options = {
        'small': {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, None]
        },
        'medium': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, None]
        },
        'large': {
            'n_estimators': [200, 300, 500],
            'max_depth': [15, 20, None]
        }
    }
    return options[size]

def extract_section(code, section_name):
    pattern = rf'# === {section_name}_START ===\n(.*?)\n# === {section_name}_END ==='
    match = re.search(pattern, code, re.DOTALL)
    return match.group(1).strip() if match else None

def replace_section(code, section_name, new_content):
    pattern = rf'# === {section_name}_START ===\n(.*?)\n# === {section_name}_END ==='
    replacement = f'# === {section_name}_START ===\n{new_content}\n# === {section_name}_END ==='
    return re.sub(pattern, replacement, code, flags=re.DOTALL)

def change_hyperparameter_tuning(code):
    model_name = extract_algorithm_from_code(code)
    current_hyperparameters = extract_hyperparameters_from_code(code)

    param_spaces = {
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 15, 20],
            'min_samples_split': [2, 5, 10]
        },
        'GradientBoostingClassifier': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    }

    if model_name not in param_spaces:
        return code

    model_params = param_spaces[model_name]
    new_params = {}

    for param, options in model_params.items():
        current_val = current_hyperparameters.get(param, options[0])

        try:
            current_idx = options.index(current_val)
            next_idx = (current_idx + 1) % len(options)
            new_params[param] = options[next_idx]
        except ValueError:
            new_params[param] = options[0]

    param_strings = [f'{k}={repr(v)}' for k, v in new_params.items()]
    if 'random_state' not in new_params:
        param_strings.append('random_state=42')

    new_model_code = f"model = {model_name}({','.join(param_strings)})"
    return new_model_code

def change_algorithm(code):
    model_name = extract_algorithm_from_code(code)
    model_names = ['RandomForestClassifier', 'GradientBoostingClassifier']
    current_idx = model_names.index(model_name)
    next_idx = (current_idx + 1) % len(model_names)
    next_model = model_names[next_idx]
    print(f'Current model: {model_name}, next model: {next_model}')
    default_params = {
        'RandomForestClassifier': 'n_estimators=100, random_state=42',
        'GradientBoostingClassifier': 'n_estimators=100, learning_rate=0.1, random_state=42'
    }
    new_model_line = f"model = {next_model}({default_params[next_model]})"
    return new_model_line


def change_imports(new_model_name):
    import_mapping = {
        'RandomForestClassifier': 'from sklearn.ensemble import RandomForestClassifier',
        'GradientBoostingClassifier': 'from sklearn.ensemble import GradientBoostingClassifier',
        'LogisticRegression': 'from sklearn.linear_model import LogisticRegression',
        'SVC': 'from sklearn.svm import SVC'
    }

    return import_mapping.get(new_model_name, 'from sklearn.ensemble import RandomForestClassifier')