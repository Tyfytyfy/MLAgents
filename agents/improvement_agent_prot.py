import json
import time
from datetime import datetime, timedelta

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from states.data_state import MLState, ImprovementMessage, SolutionSnapshot
from utils.extract_text import extract_algorithm_from_code, get_execution_results_from_messages
from utils.thompson_bandit import ThompsonStrategyBandit

strategy_bandit = ThompsonStrategyBandit()


def improvement_agent_node(state: MLState) -> MLState:
    print(state.get('execution_time'))
    print(state.get('total_tokens'))
    time.sleep(3)
    start = time.time()
    #llm = ChatOpenAI(model="gpt-4o", temperature=0)

    #validation_results = get_validation_results_from_messages(state.get("messages", []))
    execution_results = get_execution_results_from_messages(state.get("messages", []))

    current_iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    #improvement_history = state.get("improvement_history", [])
    solution_history = state.get("solution_history", [])
    current_best = state.get("current_best_solution")
    messages = state.get("messages", [])
    focus_count = count_focus_attempts(messages)
    print(f"Focus count: {focus_count}")

    current_accuracy = execution_results.get("model_metrics", {}).get("accuracy", 0)

    if current_iteration >= max_iterations:
        improvement_message = ImprovementMessage(
            role="assistant",
            content=f"Maximum iterations ({max_iterations}) reached. Stopping improvement attempts.",
            type="improvement",
            improvement_plan="Maximum iteration limit reached",
            root_cause_analysis="Could not achieve satisfactory results within iteration limit",
            recommended_actions=["Manual review required"],
            restart_point="END",
            change_focus="none"
        )

        existing_messages = state.get("messages", [])
        end = time.time()
        return create_return_state(state, existing_messages + [improvement_message], solution_history, current_best, response=None, duration=end-start)

    next_strategy = strategy_bandit.select_strategy()

    if current_iteration > 0:
        best_accuracy = get_best_accuracy_from_history(solution_history)
        print(f"DEBUG: Current: {current_accuracy}, Best from history: {best_accuracy}")
        executed_strategy = state.get("change_focus")

        if current_accuracy > 80:
            reward = 1.0
            print(f"HIGH PERFORMANCE: {executed_strategy} gets reward +1.0")
        elif current_accuracy > best_accuracy:
            reward = 1.0
            print(f"NEW BEST: {executed_strategy} gets reward +1.0")
        elif current_accuracy == best_accuracy:
            reward = 0.3
            print(f"EQUAL: {executed_strategy} gets reward +0.3")
        elif current_accuracy >= best_accuracy - 3:
            reward = 0.0
            print(f"CLOSE BUT WORSE: {executed_strategy} gets reward 0.0")
        else:
            reward = 0.0
            print(f"POOR: {executed_strategy} gets reward 0.0")

        strategy_bandit.update(executed_strategy, reward)
        print(f"Updated bandit: {executed_strategy} with reward {reward}")
        print(f'Alphas: {strategy_bandit.alpha}')
        print(f'Betas: {strategy_bandit.beta}')

    improvement_message = ImprovementMessage(
        role="assistant",
        content=f"Bandit selected strategy: {next_strategy}",
        type="improvement",
        improvement_plan=f"Selected {next_strategy} using Thompson Sampling",
        root_cause_analysis=f"Bandit algorithm chose {next_strategy} based on historical performance",
        recommended_actions=[f"Execute {next_strategy} strategy"],
        restart_point=get_restart_point_for_strategy(next_strategy),
        change_focus=next_strategy
    )

    existing_messages = state.get("messages", [])
    end = time.time()
    return create_return_state(state, existing_messages + [improvement_message], solution_history, current_best,
                               response=None, duration=end-start, change_focus=next_strategy)


def create_solution_snapshot(state: MLState, iteration: int, accuracy: float,
                             execution_results: dict) -> SolutionSnapshot:
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

    return SolutionSnapshot(
        iteration=iteration,
        accuracy=accuracy,
        model_metrics=execution_results.get("model_metrics", {}),
        data_analysis=state.get("data_analysis", {}),
        target_analysis=state.get("target_analysis", {}),
        feature_engineering=state.get("feature_engineering", {}),
        generated_code=generated_code,
        status="success" if accuracy > 0 else "failed",
        timestamp=datetime.now().isoformat()
    )


def build_solution_history_context(solution_history: list, current_best: SolutionSnapshot, current_accuracy: float) -> str:
    if not solution_history:
        return "No previous solutions to compare with. This is the first iteration."

    context_parts = []
    context_parts.append("SOLUTION PERFORMANCE HISTORY:")

    for solution in solution_history:
        status_marker = "BEST" if current_best and solution['iteration'] == current_best['iteration'] else "    "
        context_parts.append(
            f"{status_marker} Iteration {solution['iteration']}: {solution['accuracy']:.3f} accuracy - {solution['status']}")

    if current_best:
        context_parts.append(f"\nBEST SOLUTION SO FAR:")
        context_parts.append(f"- Accuracy: {current_best['accuracy']:.3f}")
        context_parts.append(f"- Iteration: {current_best['iteration']}")
        context_parts.append(f"- Algorithm: {extract_algorithm_from_code(current_best.get('generated_code', ''))}")

        if current_accuracy > 0:
            improvement = current_accuracy - current_best["accuracy"]
            if improvement > 0.02:
                context_parts.append(
                    f"- CURRENT RESULT: {current_accuracy:.3f} (+{improvement:.3f}) - SIGNIFICANT IMPROVEMENT!")
            elif improvement > 0:
                context_parts.append(
                    f"- CURRENT RESULT: {current_accuracy:.3f} (+{improvement:.3f}) - marginal improvement")
            elif improvement < -0.02:
                context_parts.append(
                    f"- CURRENT RESULT: {current_accuracy:.3f} ({improvement:.3f}) - SIGNIFICANT DEGRADATION")
            else:
                context_parts.append(
                    f"- CURRENT RESULT: {current_accuracy:.3f} ({improvement:.3f}) - similar performance")
    else:
        context_parts.append("\nNo successful solutions yet.")

    return "\n".join(context_parts)





def create_return_state(state: MLState, messages: list, solution_history: list, best_solution: SolutionSnapshot,
                        response, duration, improvement_history: list = None, change_focus: str = None) -> MLState:
    if improvement_history is None:
        improvement_history = state.get("improvement_history", [])

    print(f"=== CREATE RETURN STATE DEBUG ===")
    print(f"Solution history length: {len(solution_history)}")
    print(f"Best solution: {best_solution.get('accuracy', 'None') if best_solution else 'None'}")
    print(f"Change focus: {change_focus}")
    print("=== END RETURN STATE DEBUG ===")
    total_tokens = 0
    if response is not None:
        total_tokens = response.usage_metadata.get('total_tokens', 0)
    return {
        "data_file_path": state["data_file_path"],
        "data_summary": state["data_summary"],
        "user_objective": state["user_objective"],
        "iteration_count": state.get("iteration_count", 0) + 1,
        "max_iterations": state.get("max_iterations", 3),
        "improvement_history": improvement_history,
        "messages": messages,
        "data_analysis": state.get("data_analysis", {}),
        "target_analysis": state.get("target_analysis", {}),
        "feature_engineering": state.get("feature_engineering", {}),
        "solution_history": solution_history,
        "current_best_solution": best_solution,
        "change_focus": change_focus,
        "execution_time": state.get("execution_time", timedelta(0)) + timedelta(seconds=duration),
        "total_tokens": state.get("total_tokens", 0) + total_tokens
    }


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
                "improvement_suggestions": msg_data.get("improvement_suggestions", []),
                "performance_analysis": msg_data.get("performance_analysis", {}),
                "requirements_met": msg_data.get("requirements_met", {})
            }
    return {}

def count_focus_attempts(messages):
    focus_counts = {}
    for message in messages:
        if hasattr(message, 'additional_kwargs') and message.additional_kwargs.get("change_focus"):
            focus = message.additional_kwargs.get("change_focus", "")
            focus_counts[focus] = focus_counts.get(focus, 0) + 1
    return focus_counts


def get_previous_accuracy_from_history(solution_history):
    if len(solution_history) < 2:
        return 0
    return solution_history[-2].get('accuracy', 0)

def get_previous_strategy_from_messages(messages):
    for message in reversed(messages[:-4]):
        if hasattr(message, 'additional_kwargs') and message.additional_kwargs.get("type") == "improvement":
            return message.additional_kwargs.get("change_focus", "HYPERPARAMETERS")
    return "HYPERPARAMETERS"

def get_restart_point_for_strategy(strategy):
    mapping = {
        'HYPERPARAMETERS': 'code_generator_agent',
        'ALGORITHM_SELECTION': 'code_generator_agent',
        'feature_selection': 'feature_agent',
        'scaling_method': 'feature_agent',
        'encoding_method': 'feature_agent',
        'class_balancing': 'feature_agent'
    }
    return mapping.get(strategy, 'code_generator_agent')

def get_best_accuracy_from_history(solution_history):
    if not solution_history:
        return 0

    best_accuracy = 0
    for solution in solution_history:
        accuracy = solution.get("accuracy", 0)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
    return best_accuracy

