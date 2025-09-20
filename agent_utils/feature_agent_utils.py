from datetime import timedelta

from states.data_state import FeatureEngineeringMessage, MLState


def no_focus_change_feature_agent_prompt(user_objective: str, data_analysis: dict, target_analysis: dict, data_for_analysis: str) -> str:
    return f"""
        Plan feature engineering for this machine learning project:
        
        User objective: {user_objective}
        Data profile: {data_analysis.get('profile', '')}
        Available columns: {data_analysis.get('columns', [])}
        Data types: {data_analysis.get('types', '')}
        Missing data info: {data_analysis.get('missing', '')}
        Target variable: {target_analysis.get('target_variable', '')}
        Problem type: {target_analysis.get('problem_type', '')}
        
        Dataset details:
        {data_for_analysis}
        
        Perform:
        1. Propose preprocessing that supports user objective
        2. Choose encoding method for categorical variables
        3. Choose scaling method for numerical variables
        4. Identify features to drop
        
        Return ONLY valid JSON:
        {{
            "feature_plan": "overall strategy",
            "preprocessing_steps": ["step1", "step2", "step3"],
            "encoding_method": "specific encoding method",
            "scaling_method": "specific scaling method", 
            "features_to_drop": ["feature1", "feature2"]
        }}
        """

def focus_change_feature_agent_prompt(user_objective: str, data_analysis: dict, target_analysis: dict, data_for_analysis: str, previous_approach: str, change_focus: str) -> str:
    if change_focus == "encoding_method":
        prompt = f"""
            User objective: {user_objective}
            Data profile: {data_analysis.get('profile', '')}
            Available columns: {data_analysis.get('columns', [])}
            Data types: {data_analysis.get('types', '')}
            Missing data info: {data_analysis.get('missing', '')}
            Target variable: {target_analysis.get('target_variable', '')}
            Problem type: {target_analysis.get('problem_type', '')}

            Current encoding method: {previous_approach}

            Dataset details:
            {data_for_analysis}

            Based on data analysis, choose different encoding method for better performance.

            Return ONLY valid JSON:
            {{
                "encoding_method": "new specific encoding method"
            }}
            """
    elif change_focus == "scaling_method":
        prompt = f"""
            User objective: {user_objective}
            Data profile: {data_analysis.get('profile', '')}
            Available columns: {data_analysis.get('columns', [])}
            Data types: {data_analysis.get('types', '')}
            Missing data info: {data_analysis.get('missing', '')}
            Target variable: {target_analysis.get('target_variable', '')}
            Problem type: {target_analysis.get('problem_type', '')}

             Current scaling method: {previous_approach}

            Dataset details:
            {data_for_analysis}

            Based on data analysis, choose different scaling_method for better performance.

            Return ONLY valid JSON:
            {{
                "scaling_method": "new specific scaling method"
            }}
            """
    elif change_focus == "feature_selection":
        prompt =  f"""
            User objective: {user_objective}
            Data profile: {data_analysis.get('profile', '')}
            Available columns: {data_analysis.get('columns', [])}
            Data types: {data_analysis.get('types', '')}
            Missing data info: {data_analysis.get('missing', '')}
            Target variable: {target_analysis.get('target_variable', '')}
            Problem type: {target_analysis.get('problem_type', '')}
            
            Current features to drop: {previous_approach}
            
            Dataset details:
            {data_for_analysis}
            
            Based on data analysis, choose different features to drop for better performance.
            
            Return ONLY valid JSON:
            {{
                "features_to_drop": ["feature1", "feature2"]
            }}
            """
    elif change_focus == "class_balancing":
        prompt = f"""
            User objective: {user_objective}
            Data profile: {data_analysis.get('profile', '')}
            Available columns: {data_analysis.get('columns', [])}
            Data types: {data_analysis.get('types', '')}
            Missing data info: {data_analysis.get('missing', '')}
            Target variable: {target_analysis.get('target_variable', '')}
            Problem type: {target_analysis.get('problem_type', '')}

            Current preprocessing: {previous_approach}

            Dataset details:
            {data_for_analysis}

            Add class balancing technique (SMOTE, undersampling, class weights).
            Keep other preprocessing steps identical.

            Return ONLY valid JSON:
            {{
                "preprocessing_steps": ["new_step1", "new_step2" "new_step3"]
            }}
            """
    else:
        prompt = no_focus_change_feature_agent_prompt(user_objective, data_analysis, target_analysis, data_for_analysis)

    return prompt

def create_feature_message_from_config(updated_config: dict, content_text: str):
    return FeatureEngineeringMessage(
                role="assistant",
                content=content_text,
                type="feature_engineering",
                feature_plan=updated_config.get('plan', ''),
                preprocessing_steps=updated_config.get('preprocessing_steps', []),
                encoding_method=updated_config.get('encoding_method', ''),
                scaling_method=updated_config.get('scaling_method', ''),
                features_to_drop=updated_config.get("features_to_drop", [])
            )

def create_feature_return_state(state: MLState, feature_message: FeatureEngineeringMessage, response, duration) -> MLState:
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
        "messages": state.get('messages', []) + [feature_message],
        "data_analysis": state.get("data_analysis", {}),
        "target_analysis": state.get("target_analysis", {}),
        "feature_engineering": {
            "plan": feature_message["feature_plan"],
            "preprocessing_steps": feature_message["preprocessing_steps"],
            "encoding_method": feature_message["encoding_method"],
            "scaling_method": feature_message["scaling_method"],
            "features_to_drop": feature_message["features_to_drop"],
        },
        "solution_history": state.get("solution_history", []),
        "current_best_solution": state.get("current_best_solution"),
        "change_focus": state.get("change_focus"),
        "execution_time": state.get("execution_time", timedelta(0)) + timedelta(seconds=duration),
        "total_tokens": state.get("total_tokens", 0) + total_tokens
    }