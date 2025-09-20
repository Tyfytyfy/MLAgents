from datetime import timedelta

from langgraph.constants import END
from langgraph.graph import StateGraph

from agents.code_execution_agent_prot import code_executor_agent_node
from agents.explanation_agent_prot import explanation_agent_node
from agents.improvement_agent_prot import improvement_agent_node
from agents.validation_agent_prot import validation_agent_node
from states.data_state import MLState
from agents.data_agent_prot import data_agent_node
from agents.target_agent_prot import target_agent_node
from agents.feature_agent_prot import feature_agent_node
from agents.code_generator_agent_prot import code_generator_agent_node


def create_ml_workflow():
    workflow = StateGraph(MLState)

    workflow.add_node('data_agent', data_agent_node)
    workflow.add_node('target_agent', target_agent_node)
    workflow.add_node('feature_agent', feature_agent_node)
    workflow.add_node('code_generator_agent', code_generator_agent_node)
    workflow.add_node('code_executor_agent', code_executor_agent_node)
    workflow.add_node('validation_agent', validation_agent_node)
    workflow.add_node('explanation_agent', explanation_agent_node)
    workflow.add_node('improvement_agent', improvement_agent_node)

    workflow.set_entry_point('data_agent')
    workflow.add_edge('data_agent', 'target_agent')
    workflow.add_edge('target_agent', 'feature_agent')
    workflow.add_edge('feature_agent', 'code_generator_agent')
    workflow.add_edge('code_generator_agent', 'code_executor_agent')
    workflow.add_edge('code_executor_agent', 'validation_agent')

    workflow.add_conditional_edges('validation_agent', route_after_validation, {
        'explanation_agent': 'explanation_agent',
        'improvement_agent': 'improvement_agent',
        END: END
    })

    workflow.add_conditional_edges(
        'improvement_agent',
        route_after_improvement,
        {
            'data_agent': 'data_agent',
            'target_agent': 'target_agent',
            'feature_agent': 'feature_agent',
            'code_generator_agent': 'code_generator_agent',
            END: END
        }
    )
    workflow.add_edge('explanation_agent', END)

    return workflow.compile()


def route_after_validation(state: MLState) -> str:
    messages = state.get("messages", [])

    print("=== DEBUG: VALIDATION ROUTING ===")
    print(f"Total messages: {len(messages)}")

    for message in reversed(messages):
        print(f"Checking message type: {type(message)}")

        if hasattr(message, 'additional_kwargs') and message.additional_kwargs.get("type") == "validation":
            status = message.additional_kwargs.get("validation_status", "")
            print(f"Found validation status (additional_kwargs): {status}")
            if status == "passed":
                print("→ Routing to explanation_agent")
                return "explanation_agent"
            elif status in ["failed", "needs_improvement"]:
                print("→ Routing to improvement_agent")
                return "improvement_agent"
            else:
                print(f"→ Unknown status '{status}', ending")
                return END

        elif isinstance(message, dict) and message.get("type") == "validation":
            status = message.get("validation_status", "")
            print(f"Found validation status (dict): {status}")
            if status == "passed":
                print("→ Routing to explanation_agent")
                return "explanation_agent"
            elif status in ["failed", "needs_improvement"]:
                print("→ Routing to improvement_agent")
                return "improvement_agent"
            else:
                print(f"→ Unknown status '{status}', ending")
                return END

    print("→ No validation message found, ending")
    print("=== END DEBUG ===")
    return END


def route_after_improvement(state: MLState) -> str:
    messages = state.get("messages", [])
    current_iteration = state.get("iteration_count", 0)

    print("=== DEBUG: IMPROVEMENT ROUTING ===")
    print(f"Current iteration: {current_iteration}")

    if current_iteration >= state.get("max_iterations", 3):
        print("Maximum iterations reached, ending")
        return END

    for message in reversed(messages):
        if hasattr(message, 'additional_kwargs') and message.additional_kwargs.get("type") == "improvement":
            restart_point = message.additional_kwargs.get("restart_point", "END")
            print(f"Found restart point (additional_kwargs): {restart_point}")
            return restart_point
        elif isinstance(message, dict) and message.get("type") == "improvement":
            restart_point = message.get("restart_point", "END")
            print(f"Found restart point (dict): {restart_point}")
            return restart_point

    print("No improvement message found, ending")
    print("=== END DEBUG ===")
    return END


def get_analysis_results(final_state: MLState) -> dict:
    return {
        "data_analysis": final_state.get("data_analysis", {}),
        "target_analysis": final_state.get("target_analysis", {}),
        "feature_engineering": final_state.get("feature_engineering", {}),
        "messages": final_state.get("messages", []),
        "solution_history": final_state.get("solution_history", []),
        "current_best_solution": final_state.get("current_best_solution"),
        "total_tokens": final_state.get("total_tokens", 0),
        "execution_time": final_state.get("execution_time", timedelta(0))
    }


def get_generated_code(final_state: MLState) -> dict:
    messages = final_state.get("messages", [])

    for message in reversed(messages):
        if hasattr(message, 'additional_kwargs') and message.additional_kwargs.get("type") == "code_generation":
            return {
                "generated_code": message.additional_kwargs.get("generated_code", ""),
                "pipeline_code": message.additional_kwargs.get("pipeline_code", ""),
                "model_code": message.additional_kwargs.get("model_code", "")
            }
        elif isinstance(message, dict) and message.get("type") == "code_generation":
            return {
                "generated_code": message.get("generated_code", ""),
                "pipeline_code": message.get("pipeline_code", ""),
                "model_code": message.get("model_code", "")
            }

    return {
        "generated_code": "",
        "pipeline_code": "",
        "model_code": ""
    }


def get_execution_results(final_state: MLState) -> dict:
    messages = final_state.get("messages", [])

    for message in reversed(messages):
        if hasattr(message, 'content'):
            if hasattr(message, 'additional_kwargs') and message.additional_kwargs.get("type") == "code_execution":
                return {
                    "execution_status": message.additional_kwargs.get("execution_status", ""),
                    "model_metrics": message.additional_kwargs.get("model_metrics", {}),
                    "execution_logs": message.additional_kwargs.get("execution_logs", ""),
                    "error_details": message.additional_kwargs.get("error_details", ""),
                    "saved_model_path": message.additional_kwargs.get("saved_model_path", "")
                }
        elif isinstance(message, dict) and message.get("type") == "code_execution":
            return {
                "execution_status": message.get("execution_status", ""),
                "model_metrics": message.get("model_metrics", {}),
                "execution_logs": message.get("execution_logs", ""),
                "error_details": message.get("error_details", ""),
                "saved_model_path": message.get("saved_model_path", "")
            }

    return {
        "execution_status": "",
        "model_metrics": {},
        "execution_logs": "",
        "error_details": "",
        "saved_model_path": ""
    }


def get_validation_results(final_state: MLState) -> dict:
    messages = final_state.get("messages", [])

    for message in reversed(messages):
        if hasattr(message, 'content'):
            if hasattr(message, 'additional_kwargs') and message.additional_kwargs.get("type") == "validation":
                return {
                    "validation_status": message.additional_kwargs.get("validation_status", ""),
                    "model_assessment": message.additional_kwargs.get("model_assessment", ""),
                    "improvement_suggestions": message.additional_kwargs.get("improvement_suggestions", []),
                    "performance_analysis": message.additional_kwargs.get("performance_analysis", {}),
                    "requirements_met": message.additional_kwargs.get("requirements_met", {})
                }
        elif isinstance(message, dict) and message.get("type") == "validation":
            return {
                "validation_status": message.get("validation_status", ""),
                "model_assessment": message.get("model_assessment", ""),
                "improvement_suggestions": message.get("improvement_suggestions", []),
                "performance_analysis": message.get("performance_analysis", {}),
                "requirements_met": message.get("requirements_met", {})
            }

    return {
        "validation_status": "",
        "model_assessment": "",
        "improvement_suggestions": [],
        "performance_analysis": {},
        "requirements_met": {}
    }


def get_explanation_results(final_state: MLState) -> dict:
    messages = final_state.get("messages", [])

    for message in reversed(messages):
        if hasattr(message, 'content'):
            if hasattr(message, 'additional_kwargs') and message.additional_kwargs.get("type") == "explanation":
                return {
                    "feature_importance": message.additional_kwargs.get("feature_importance", {}),
                    "model_explanation": message.additional_kwargs.get("model_explanation", ""),
                    "domain_validation": message.additional_kwargs.get("domain_validation", "")
                }
        elif isinstance(message, dict) and message.get("type") == "explanation":
            return {
                "feature_importance": message.get("feature_importance", {}),
                "model_explanation": message.get("model_explanation", ""),
                "domain_validation": message.get("domain_validation", "")
            }

    return {
        "feature_importance": {},
        "model_explanation": "",
        "domain_validation": ""
    }


def get_improvement_results(final_state: MLState) -> dict:
    messages = final_state.get("messages", [])
    improvement_attempts = []

    for message in messages:
        if hasattr(message, 'content'):
            if hasattr(message, 'additional_kwargs') and message.additional_kwargs.get("type") == "improvement":
                improvement_attempts.append({
                    "improvement_plan": message.additional_kwargs.get("improvement_plan", ""),
                    "root_cause_analysis": message.additional_kwargs.get("root_cause_analysis", ""),
                    "recommended_actions": message.additional_kwargs.get("recommended_actions", []),
                    "restart_point": message.additional_kwargs.get("restart_point", ""),
                    "change_focus": message.additional_kwargs.get("change_focus", "")
                })
        elif isinstance(message, dict) and message.get("type") == "improvement":
            improvement_attempts.append({
                "improvement_plan": message.get("improvement_plan", ""),
                "root_cause_analysis": message.get("root_cause_analysis", ""),
                "recommended_actions": message.get("recommended_actions", []),
                "restart_point": message.get("restart_point", ""),
                "change_focus": message.get("change_focus", "")
            })

    return {
        "improvement_attempts": improvement_attempts,
        "total_iterations": final_state.get("iteration_count", 0),
        "max_iterations": final_state.get("max_iterations", 3),
        "improvement_history": final_state.get("improvement_history", []),
        "solution_history": final_state.get("solution_history", []),
        "current_best_solution": final_state.get("current_best_solution")
    }