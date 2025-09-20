from workflow import get_execution_results


def display_solution_tracking_details(analysis_results, result):
    print("\n" + "=" * 60)
    print("SOLUTION TRACKING & PERFORMANCE HISTORY")
    print("=" * 60)

    solution_history = analysis_results.get('solution_history', [])
    current_best = analysis_results.get('current_best_solution')

    if not solution_history:
        print("No solution history available - this indicates first successful run or no iterations needed.")

        final_execution = get_execution_results(result)
        if final_execution.get('model_metrics', {}).get('accuracy'):
            print(f"\nFINAL MODEL PERFORMANCE:")
            print(f"   Accuracy: {final_execution['model_metrics']['accuracy']:.2f}%")
            print(f"   Status: {final_execution.get('execution_status', 'unknown')}")
        return

    print(f"\nSOLUTION PERFORMANCE TIMELINE:")
    for solution in solution_history:
        status_marker = "BEST" if current_best and solution['iteration'] == current_best['iteration'] else "    "
        print(
            f"{status_marker} Iteration {solution['iteration']}: {solution['accuracy']:.3f} accuracy - {solution['status']} ({solution['timestamp'][:19]})")

    if current_best:
        print(f"\nBEST SOLUTION DETAILS:")
        print(f"   Accuracy: {current_best['accuracy']:.3f}")
        print(f"   Iteration: {current_best['iteration']}")
        print(f"   Timestamp: {current_best['timestamp'][:19]}")
        print(f"   Status: {current_best['status']}")

        if 'model_metrics' in current_best and current_best['model_metrics']:
            print(f"   Model Metrics: {current_best['model_metrics']}")

    print("\n" + "=" * 60)


def display_improvement_details(improvement_results):
    print("\n" + "=" * 50)
    print("IMPROVEMENT & ITERATION DETAILS")
    print("=" * 50)

    print(f"Total Iterations: {improvement_results['total_iterations']}")
    print(f"Max Iterations: {improvement_results['max_iterations']}")

    if improvement_results['improvement_history']:
        print(f"\nImprovement History:")
        for i, history_item in enumerate(improvement_results['improvement_history'], 1):
            print(f"  {i}. {history_item}")

    if improvement_results['improvement_attempts']:
        print(f"\nDetailed Improvement Attempts:")
        for i, attempt in enumerate(improvement_results['improvement_attempts'], 1):
            print(f"\n--- Attempt {i} ---")
            print("-" * 50)
            print(f"Change Focus: {attempt.get('change_focus', 'N/A')}")
            print("-" * 50)
            print(f"Root Cause: {attempt['root_cause_analysis']}")
            print(f"Improvement Plan: {attempt['improvement_plan']}")
            print(f"Recommended Actions:")
            for j, action in enumerate(attempt['recommended_actions'], 1):
                print(f"  {j}. {action}")
            print(f"Restart Point: {attempt['restart_point']}")
    else:
        print("\nNo improvement attempts needed - model succeeded on first try!")

    print("=" * 50)


def display_data_analysis_results(data_analysis: dict):
    print(f"Profile: {data_analysis.get('profile', '')}")
    print(f"Columns: {data_analysis.get('columns', [])}")
    print(f"Types: {data_analysis.get('types', '')}")
    print(f"Missing: {data_analysis.get('missing', '')}")


def display_target_analysis_results(target_analysis: dict):
    print(f"Target Variable: {target_analysis.get('target_variable', '')}")
    print(f"Problem Type: {target_analysis.get('problem_type', '')}")
    print(f"Target Analysis: {target_analysis.get('analysis', '')}")


def display_feature_engineering_results(feature_engineering: dict):
    print(f"Feature Plan: {feature_engineering.get('plan', '')}")
    print(f"Preprocessing Steps: {feature_engineering.get('preprocessing_steps', [])}")
    print(f"Encoding Method: {feature_engineering.get('encoding_method', '')}")
    print(f"Scaling Method: {feature_engineering.get('scaling_method', '')}")
    print(f"Features to Drop: {feature_engineering.get('features_to_drop', [])}")


def display_validation_results(validation_dict: dict):
    print(f"\n=== MODEL VALIDATION RESULTS ===")
    print(f"Validation Status: {validation_dict['validation_status']}")
    print(f"Requirements Met: {validation_dict['requirements_met']}")
    print(f"Assessment: {validation_dict['model_assessment']}")
    print(f"Suggestions: {validation_dict['improvement_suggestions']}")


def display_explanation_results(explanation_dict: dict):
    print(f"\n=== MODEL EXPLANATION RESULTS ===")
    print(f"Feature Importance: {explanation_dict['feature_importance']}")
    print(f"Model Explanation: {explanation_dict['model_explanation']}")
    print(f"Domain Validation: {explanation_dict['domain_validation']}")