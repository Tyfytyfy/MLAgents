from datetime import timedelta

from states.data_state import MLState, ExplanationMessage


def create_explanation_return_state(state: MLState, message: ExplanationMessage, response, duration) -> MLState:
    total_tokens = 0
    if response is not None:
        total_tokens = response.usage_metadata.get('total_tokens', 0)
    return {
        "data_file_path": state["data_file_path"],
        "data_summary": state["data_summary"],
        "user_objective": state["user_objective"],
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 3),
        "improvement_history": state.get("improvement_history", []),
        "messages": state.get("messages", []) + [message],
        "data_analysis": state.get("data_analysis", {}),
        "target_analysis": state.get("target_analysis", {}),
        "feature_engineering": state.get("feature_engineering", {}),
        "solution_history": state.get("solution_history", []),
        "current_best_solution": state.get("current_best_solution"),
        "change_focus": state.get("change_focus"),
        "execution_time": state.get("execution_time", timedelta(0)) + timedelta(seconds=duration),
        "total_tokens": state.get("total_tokens", 0) + total_tokens
    }

def create_passed_explanation_state(state: MLState, response, duration) -> MLState:
    total_tokens = 0
    if response is not None:
        total_tokens = response.usage_metadata.get('total_tokens', 0)
    return {
        "data_file_path": state["data_file_path"],
        "data_summary": state["data_summary"],
        "user_objective": state["user_objective"],
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 3),
        "improvement_history": state.get("improvement_history", []),
        "messages": state.get("messages", []),
        "data_analysis": state.get("data_analysis", {}),
        "target_analysis": state.get("target_analysis", {}),
        "feature_engineering": state.get("feature_engineering", {}),
        "solution_history": [],
        "current_best_solution": None,
        "change_focus": None,
        "execution_time": state.get("execution_time", timedelta(0)) + timedelta(seconds=duration),
        "total_tokens": state.get("total_tokens", 0) + total_tokens
    }

def create_explanation_agent_prompt(state: MLState, model_type: str, validation_results: dict, execution_results: dict,
                                    feature_importance: dict) -> str:
    return  f"""
        Explain this successful ML model using REAL feature importance data extracted from the trained model:

        User Objective: {state['user_objective']}

        CONFIRMED MODEL TYPE: {model_type}
        (This is the ACTUAL model that was trained, not a guess)

        Model Performance:
        - Validation Status: {validation_results.get('validation_status', '')}
        - Model Metrics: {execution_results.get('model_metrics', {})}
        - Model Assessment: {validation_results.get('model_assessment', '')}

        REAL FEATURE IMPORTANCE (extracted from the {model_type} model):
        {feature_importance}

        Dataset Context:
        - Problem Type: {state.get('target_analysis', {}).get('problem_type', '')}
        - Target Variable: {state.get('target_analysis', {}).get('target_variable', '')}
        - Features: {state.get('data_analysis', {}).get('columns', [])}
        - Data Types: {state.get('data_analysis', {}).get('types', '')}
        - Feature Engineering Plan: {state.get('feature_engineering', {}).get('plan', '')}
        - Encoding Method: {state.get('feature_engineering', {}).get('encoding_method', '')}
        - Scaling Method: {state.get('feature_engineering', {}).get('scaling_method', '')}

        Execution Results:
        - Execution Logs: {execution_results.get('execution_logs', '')}

        Provide comprehensive model explanation using the CONFIRMED model type and REAL feature importance:
        1. Use the actual feature importance scores extracted from the trained {model_type} model
        2. Explain how {model_type} specifically works and why these importance scores make sense for this model type
        3. Validate the importance ranking against domain knowledge and scientific understanding
        4. Do NOT guess or speculate - use only confirmed information

        Return ONLY valid JSON in this exact format:
        {{
            "feature_importance": {feature_importance if feature_importance else {"note": "No real feature importance available"} },
            "model_explanation": "detailed explanation of how {model_type} specifically works, why it's suitable for this problem, and how it generates these exact importance scores",
            "domain_validation": "analysis of whether the REAL feature importance from {model_type} makes sense from a domain/scientific perspective"
        }}

        IMPORTANT: Use ONLY confirmed information. The model type is {model_type} - do not use words like 'likely', 'probably', 'appears to be'.
        """