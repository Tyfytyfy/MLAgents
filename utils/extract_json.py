import json

from langchain_core.messages import HumanMessage


def extract_json_from_output(result: str):
    if "```json" in result:
        json_str = result.split("```json")[1].split("```")[0].strip()
    else:
        json_str = result

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw JSON string: {json_str}")

        clean_json = clean_json_response(json_str)
        print(f"Cleaned JSON: {clean_json}")

        try:
            return json.loads(clean_json)
        except json.JSONDecodeError as e2:
            print(f"Still failed after cleaning: {e2}")
            return {}

def clean_json_response(json_str: str) -> str:
    import re

    json_str = re.sub(r'//.*', '', json_str)
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    json_str = re.sub(r'\s+', ' ', json_str)
    json_str = json_str.strip()

    return json_str

def extract_requirements_with_llm(user_objective: str, llm) -> dict:
    requirements_prompt = f"""
    Extract specific ML model requirements from this user objective:

    User Objective: {user_objective}

    Identify and extract any mentioned requirements such as:
    - Minimum accuracy/performance thresholds
    - Maximum training time or speed requirements  
    - Model explainability/interpretability needs
    - Model size or resource constraints
    - Custom metrics (R², MAE, precision, recall, etc.)
    - Business constraints or domain-specific needs

    Return ONLY valid JSON in this exact format:
    {{
        "min_accuracy": 0.7,
        "max_training_time_seconds": 300,
        "explainable": false,
        "model_size_limit_mb": null,
        "custom_metrics": {{
            "min_r2": null,
            "max_mae": null,
            "min_precision": null,
            "min_recall": null
        }},
        "business_constraints": ["constraint1", "constraint2"],
        "other_requirements": ["requirement1", "requirement2"]
    }}

    If no specific requirement is mentioned, use reasonable defaults for the domain.
    Set null for requirements that are not specified.
    """

    try:
        response = llm.invoke([HumanMessage(content=requirements_prompt)])
        result = response.content

        if "```json" in result:
            json_str = result.split("```json")[1].split("```")[0].strip()
        else:
            json_str = result

        return json.loads(json_str)

    except Exception as e:
        print(f"Error extracting requirements: {e}")
        return {
            "min_accuracy": 0.7,
            "max_training_time_seconds": 300,
            "explainable": False,
            "model_size_limit_mb": None,
            "custom_metrics": {},
            "business_constraints": [],
            "other_requirements": []
        }