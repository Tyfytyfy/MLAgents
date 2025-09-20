def target_agent_prompt(user_objective: str, data_analysis: dict, data_for_analysis: str) -> str:
    return f"""
    User objective: {user_objective}

    Analyze this data for target variable selection:

    Data profile: {data_analysis.get('profile', '')}
    Available columns: {data_analysis.get('columns', [])}
    Data types: {data_analysis.get('types', '')}
    Missing data info: {data_analysis.get('missing', '')}
    Outliers info: {data_analysis.get('outliers_info', '')}

    Dataset details:
    {data_for_analysis}

    Perform:
    1. Analyze each column as potential target variable considering user's objective
    2. Check correlations and relationships between variables
    3. Determine if this is classification or regression problem based on user goal
    4. Recommend the most likely target variable with reasoning that aligns with user objective

    Return ONLY valid JSON in this exact format:
    {{
        "target_variable": "recommended_column_name",
        "problem_type": "classification or regression",
        "target_analysis": "detailed reasoning for target selection and problem type considering user objective"
    }}
    """