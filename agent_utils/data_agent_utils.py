def data_agent_prompt(user_objective: str, data_for_analysis: str):
    return f"""
    User objective: {user_objective}

    Analyze this dataset:
    {data_for_analysis}

    Perform:
    1. Check class balance
    2. Identify NA values and outliers  
    3. Generate descriptive statistics
    4. Consider the user's goal when analyzing the dataset structure

    Return ONLY valid JSON in this exact format:
    {{
        "profile": "brief description of the dataset in context of user objective",
        "columns": ["col1", "col2", "col3"],
        "types": "description of data types",
        "missing": "missing data analysis",
        "outliers": "outlier data analysis",
    }}
    """

