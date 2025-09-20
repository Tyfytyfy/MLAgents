import json
import time
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agents.explanation_agent_prot import get_execution_results_from_messages
from agent_utils.validation_agent_utils import create_error_validation_return_state, create_validation_return_state, \
    build_performance_comparison_context
from states.data_state import MLState, ValidationMessage, SolutionSnapshot
from utils.extract_json import extract_requirements_with_llm, extract_json_from_output

load_dotenv()


def validation_agent_node(state: MLState) -> MLState:
    print(state.get('execution_time'))
    print(state.get('total_tokens'))
    time.sleep(3)
    start = time.time()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    execution_results = get_execution_results_from_messages(state.get("messages", []))
    current_best = state.get("current_best_solution")
    solution_history = state.get("solution_history", [])
    change_focus = state.get("change_focus")

    if not execution_results or execution_results.get("execution_status") != "success":
        error_message = ValidationMessage(
            role="assistant",
            content="Validation failed: No successful execution results found",
            type="validation",
            validation_status="failed",
            model_assessment="Cannot validate model - execution failed or no results found",
            improvement_suggestions=["Fix code execution errors first", "Check message passing between agents"],
            performance_analysis={},
            requirements_met={}
        )
        end = time.time()
        return create_error_validation_return_state(state, error_message, None, end-start)

    parsed_requirements = extract_requirements_with_llm(state["user_objective"], llm)
    current_accuracy = execution_results.get("model_metrics", {}).get("accuracy", 0)

    comparison_context = build_performance_comparison_context(current_best, current_accuracy, solution_history,
                                                              change_focus)

    prompt = f"""
    You are an advanced ML model validator with solution tracking capabilities.

    User Objective: {state['user_objective']}
    Extracted Requirements: {parsed_requirements}

    CURRENT EXECUTION RESULTS:
    - Status: {execution_results.get('execution_status', '')}
    - Model Metrics: {execution_results.get('model_metrics', {})}
    - Execution Logs: {execution_results.get('execution_logs', '')}

    PERFORMANCE COMPARISON CONTEXT:
    {comparison_context}

    DATA & MODEL CONTEXT:
    - Problem Type: {state.get('target_analysis', {}).get('problem_type', '')}
    - Dataset Profile: {state.get('data_analysis', {}).get('profile', '')}
    - Feature Engineering: {state.get('feature_engineering', {}).get('plan', '')}
    - Change Focus This Iteration: {change_focus or 'initial_run'}

    VALIDATION CRITERIA:
    1. Compare current performance against best historical performance
    2. Evaluate if the focused change (if any) had positive impact
    3. Check if model meets user requirements
    4. Assess model quality and potential issues
    5. Determine if this solution should become the new best solution

    Your decision logic:
    - "passed": Current model meets requirements AND (is best so far OR shows clear improvement)
    - "needs_improvement": Model works but has clear improvement potential
    - "failed": Model has serious issues or significantly worse than best solution

    Return ONLY valid JSON in this exact format:
    {{
        "validation_status": "passed | failed | needs_improvement",
        "model_assessment": "detailed assessment comparing current vs best solution and evaluating the focused change impact",
        "improvement_suggestions": ["specific suggestion based on comparison with best solution"],
        "performance_analysis": {{
            "current_vs_best": "direct comparison with best solution",
            "change_impact": "assessment of whether the focused change helped",
            "business_fit": "how well model meets business needs",
            "technical_quality": "model quality assessment"
        }},
        "requirements_met": {{
            "accuracy_requirement": true/false,
            "improvement_over_best": true/false,
            "focused_change_effective": true/false,
            "ready_for_production": true/false
        }}
    }}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content

    print("=== VALIDATION AGENT RAW RESPONSE ===")
    print(result)
    print("=== END VALIDATION RESPONSE ===")

    parsed_result = extract_json_from_output(result)

    if current_accuracy > 0 and execution_results.get("execution_status") == "success":
        generated_code = ""
        messages = state.get("messages", [])

        for message in reversed(messages):
            msg_data = None
            if hasattr(message, 'additional_kwargs'):
                msg_data = message.additional_kwargs
            elif isinstance(message, dict):
                msg_data = message

            if msg_data and msg_data.get("type") == "code_generation":
                generated_code = msg_data.get("generated_code", "")
                break

        new_snapshot = SolutionSnapshot(
            iteration=state.get("iteration_count", 0),
            accuracy=current_accuracy,
            model_metrics=execution_results.get("model_metrics", {}),
            data_analysis=state.get("data_analysis", {}),
            target_analysis=state.get("target_analysis", {}),
            feature_engineering=state.get("feature_engineering", {}),
            generated_code=generated_code,
            status="success",
            timestamp=datetime.now().isoformat()
        )

        updated_solution_history = state.get("solution_history", []).copy()
        updated_solution_history.append(new_snapshot)

        current_best = state.get("current_best_solution")
        updated_best_solution = current_best

        if not current_best or current_accuracy > current_best.get("accuracy", 0):
            updated_best_solution = new_snapshot
            print(f"=== NEW BEST SOLUTION: {current_accuracy:.2f}% ===")

        print(f"=== VALIDATION SNAPSHOT DEBUG ===")
        print(f"Created snapshot for iteration {state.get('iteration_count', 0)}")
        print(f"Current accuracy: {current_accuracy}")
        print(
            f"Best solution accuracy: {updated_best_solution.get('accuracy', 0) if updated_best_solution else 'None'}")
        print("=== END VALIDATION SNAPSHOT DEBUG ===")
    else:
        updated_solution_history = state.get("solution_history", [])
        updated_best_solution = state.get("current_best_solution")
        print("=== NO SNAPSHOT CREATED: Execution failed or no accuracy ===")

    validation_message = ValidationMessage(
        role="assistant",
        content=f"Model validation {parsed_result['validation_status']}: {parsed_result['model_assessment'][:100]}...",
        type="validation",
        validation_status=parsed_result["validation_status"],
        model_assessment=parsed_result["model_assessment"],
        improvement_suggestions=parsed_result["improvement_suggestions"],
        performance_analysis=parsed_result["performance_analysis"],
        requirements_met=parsed_result["requirements_met"]
    )

    print("=== VALIDATION AGENT RESULTS ===")
    print(f"Status: {parsed_result['validation_status']}")
    print(f"Current vs Best: {parsed_result['performance_analysis'].get('current_vs_best', 'N/A')}")
    print(f"Change Impact: {parsed_result['performance_analysis'].get('change_impact', 'N/A')}")
    print("=== END VALIDATION RESULTS ===")

    end = time.time()
    return create_validation_return_state(state, validation_message, updated_solution_history, updated_best_solution, response, (end - start))





