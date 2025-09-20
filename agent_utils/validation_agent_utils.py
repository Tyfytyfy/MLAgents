from datetime import timedelta

from states.data_state import MLState, ValidationMessage, SolutionSnapshot


def create_error_validation_return_state(state: MLState, error_message: ValidationMessage, response, duration) -> MLState:
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
        "messages": state.get("messages", []) + [error_message],
        "data_analysis": state.get("data_analysis", {}),
        "target_analysis": state.get("target_analysis", {}),
        "feature_engineering": state.get("feature_engineering", {}),
        "solution_history": state.get("solution_history", []),
        "current_best_solution": state.get("current_best_solution"),
        "change_focus": state.get("change_focus"),
        "execution_time": state.get("execution_time", timedelta(0)) + timedelta(seconds=duration),
        "total_tokens": state.get("total_tokens", 0) + total_tokens
    }

def create_validation_return_state(state: MLState, message: ValidationMessage, history, solution, response, duration) -> MLState:
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
        "solution_history": history,
        "current_best_solution": solution,
        "change_focus": state.get("change_focus"),
        "execution_time": state.get("execution_time", timedelta(0)) + timedelta(seconds=duration),
        "total_tokens": state.get("total_tokens", 0) + total_tokens
    }

def build_performance_comparison_context(current_best: SolutionSnapshot, current_accuracy: float, solution_history: list,
                                         change_focus: str) -> str:
    if not current_best:
        return f"FIRST SOLUTION: Current accuracy {current_accuracy:.3f} - no previous solutions to compare with."

    best_accuracy = current_best.get("accuracy", 0)
    improvement = current_accuracy - best_accuracy

    context_parts = []
    context_parts.append(f"PERFORMANCE COMPARISON:")
    context_parts.append(
        f"- Best Solution: {best_accuracy:.3f} accuracy (Iteration {current_best.get('iteration', 'N/A')})")
    context_parts.append(f"- Current Result: {current_accuracy:.3f} accuracy")
    context_parts.append(f"- Difference: {improvement:+.3f}")

    if change_focus:
        context_parts.append(f"- Change Focus: {change_focus}")

    if improvement > 0.02:
        context_parts.append("- ASSESSMENT: Significant improvement - this could be the new best solution")
    elif improvement > 0.005:
        context_parts.append("- ASSESSMENT: Marginal improvement - modest progress")
    elif improvement > -0.005:
        context_parts.append("- ASSESSMENT: Similar performance - no clear change")
    elif improvement > -0.02:
        context_parts.append("- ASSESSMENT: Marginal degradation - minor setback")
    else:
        context_parts.append("- ASSESSMENT: Significant degradation - this approach is not working")

    if len(solution_history) > 0:
        context_parts.append(f"\nRECENT PERFORMANCE TREND:")
        for solution in solution_history[-3:]:
            context_parts.append(f"- Iteration {solution['iteration']}: {solution['accuracy']:.3f}")

    return "\n".join(context_parts)