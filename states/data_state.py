from datetime import timedelta
from typing import TypedDict, List, Annotated, Optional, Dict, Any

from langgraph.graph import add_messages


class DataAnalysisMessage(TypedDict):
    role: str
    content: str
    type: str
    data_profile: str
    columns: List[str]
    data_types: str
    missing_data: str
    outliers_info: str
    data_length: int


class TargetAnalysisMessage(TypedDict):
    role: str
    content: str
    type: str
    target_variable: str
    problem_type: str
    target_analysis: str


class FeatureEngineeringMessage(TypedDict):
    role: str
    content: str
    type: str
    feature_plan: str
    preprocessing_steps: List[str]
    encoding_method: str
    scaling_method: str
    features_to_drop: List[str]


class CodeGenerationMessage(TypedDict):
    role: str
    content: str
    type: str
    generated_code: str
    pipeline_code: str
    model_code: str
    hyperparameters: dict


class CodeExecutionMessage(TypedDict):
    role: str
    content: str
    type: str
    execution_status: str
    model_metrics: dict
    execution_logs: str
    error_details: str
    saved_model_path: str
    feature_importance: dict


class ValidationMessage(TypedDict):
    role: str
    content: str
    type: str
    validation_status: str
    model_assessment: str
    improvement_suggestions: List[str]
    performance_analysis: dict
    requirements_met: dict


class ExplanationMessage(TypedDict):
    role: str
    content: str
    type: str
    feature_importance: dict
    model_explanation: str
    domain_validation: str


class ImprovementMessage(TypedDict):
    role: str
    content: str
    type: str
    improvement_plan: str
    root_cause_analysis: str
    recommended_actions: List[str]
    restart_point: str
    change_focus: str


class SolutionSnapshot(TypedDict):
    iteration: int
    accuracy: float
    model_metrics: Dict[str, Any]
    data_analysis: Dict[str, Any]
    target_analysis: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    generated_code: str
    status: str
    timestamp: str


class MLState(TypedDict):
    data_file_path: str
    data_summary: dict
    messages: Annotated[list, add_messages]
    iteration_count: int
    max_iterations: int
    improvement_history: List[str]
    data_analysis: dict
    target_analysis: dict
    feature_engineering: dict
    user_objective: str
    solution_history: List[SolutionSnapshot]
    current_best_solution: Optional[SolutionSnapshot]
    change_focus: Optional[str]
    execution_time: Optional[timedelta]
    total_tokens: Optional[int]