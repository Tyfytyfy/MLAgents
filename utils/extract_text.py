import re


def extract_model_type_from_code(code: str) -> str:
    model_mapping = {
        "RandomForestClassifier": "Random Forest Classifier",
        "GradientBoostingClassifier": "Gradient Boosting Classifier",
        "DecisionTreeClassifier": "Decision Tree Classifier",
        "LogisticRegression": "Logistic Regression",
        "RandomForestRegressor": "Random Forest Regressor",
        "LinearRegression": "Linear Regression",
        "SVC": "Support Vector Machine",
        "SVM": "Support Vector Machine"
    }

    for keyword, model_name in model_mapping.items():
        if keyword in code:
            return model_name

    return "Unknown Model Type"


def get_generated_code_from_messages(messages) -> dict:
    for message in reversed(messages):
        msg_data = None

        if hasattr(message, 'additional_kwargs'):
            msg_data = message.additional_kwargs
        elif isinstance(message, dict):
            msg_data = message

        if msg_data and msg_data.get("type") == "code_generation":
            return {
                "generated_code": msg_data.get("generated_code", ""),
                "pipeline_code": msg_data.get("pipeline_code", ""),
                "model_code": msg_data.get("model_code", "")
            }
    return {"generated_code": "", "pipeline_code": "", "model_code": ""}


def get_validation_results_from_messages(messages) -> dict:
    for message in reversed(messages):
        msg_data = None

        if hasattr(message, 'additional_kwargs'):
            msg_data = message.additional_kwargs
        elif isinstance(message, dict):
            msg_data = message

        if msg_data and msg_data.get("type") == "validation":
            return {
                "validation_status": msg_data.get("validation_status", ""),
                "model_assessment": msg_data.get("model_assessment", ""),
                "performance_analysis": msg_data.get("performance_analysis", {}),
                "requirements_met": msg_data.get("requirements_met", {})
            }
    return {}


def get_execution_results_from_messages(messages) -> dict:
    for message in reversed(messages):
        msg_data = None

        if hasattr(message, 'additional_kwargs'):
            msg_data = message.additional_kwargs
        elif isinstance(message, dict):
            msg_data = message

        if msg_data and msg_data.get("type") == "code_execution":
            return {
                "execution_status": msg_data.get("execution_status", ""),
                "model_metrics": msg_data.get("model_metrics", {}),
                "execution_logs": msg_data.get("execution_logs", ""),
                "error_details": msg_data.get("error_details", ""),
                "feature_importance": msg_data.get("feature_importance", {})
            }
    return {}


def extract_algorithm_from_code(code: str) -> str:
    algorithms = {
        "RandomForestClassifier": "RandomForestClassifier",
        "LogisticRegression": "LogisticRegression",
        "DecisionTreeClassifier": "DecisionTreeClassifier",
        "SVC": "SVC",
        "SVM": "SVC",
        "GradientBoostingClassifier": "GradientBoostingClassifier",
        "LinearRegression": "LinearRegression",
        "RandomForestRegressor": "RandomForestRegressor"
    }

    for keyword, model_name in algorithms.items():
        if keyword in code:
            return model_name

    return "Unknown Model Type"


def extract_hyperparameters_from_code(code: str) -> dict:
    hyperparams = {"algorithm": "unknown"}

    rf_match = re.search(r'RandomForestClassifier\((.*?)\)', code, re.DOTALL)
    if rf_match:
        hyperparams["algorithm"] = "RandomForestClassifier"
        params_str = rf_match.group(1)
        hyperparams["n_estimators"] = extract_param_value(params_str, "n_estimators", 100)
        hyperparams["max_depth"] = extract_param_value(params_str, "max_depth", None)
        hyperparams["min_samples_split"] = extract_param_value(params_str, "min_samples_split", 2)
        hyperparams["random_state"] = extract_param_value(params_str, "random_state", None)

    lr_match = re.search(r'LogisticRegression\((.*?)\)', code, re.DOTALL)
    if lr_match:
        hyperparams["algorithm"] = "LogisticRegression"
        params_str = lr_match.group(1)
        hyperparams["C"] = extract_param_value(params_str, "C", 1.0)
        hyperparams["penalty"] = extract_param_value(params_str, "penalty", "l2")
        hyperparams["solver"] = extract_param_value(params_str, "solver", "lbfgs")
        hyperparams["random_state"] = extract_param_value(params_str, "random_state", None)

    xgb_match = re.search(r'XGBClassifier\((.*?)\)', code, re.DOTALL)
    if xgb_match:
        hyperparams["algorithm"] = "XGBClassifier"
        params_str = xgb_match.group(1)
        hyperparams["n_estimators"] = extract_param_value(params_str, "n_estimators", 100)
        hyperparams["max_depth"] = extract_param_value(params_str, "max_depth", 6)
        hyperparams["learning_rate"] = extract_param_value(params_str, "learning_rate", 0.3)
        hyperparams["random_state"] = extract_param_value(params_str, "random_state", None)

    return hyperparams


def extract_param_value(params_str: str, param_name: str, default_value):
    pattern = f'{param_name}=([^,)]+)'
    match = re.search(pattern, params_str)
    if match:
        value_str = match.group(1).strip()
        if value_str == "None":
            return None
        elif value_str.isdigit():
            return int(value_str)
        elif value_str.replace(".", "").isdigit():
            return float(value_str)
        elif value_str.startswith('"') or value_str.startswith("'"):
            return value_str.strip('"\'')
        else:
            return value_str
    return default_value
