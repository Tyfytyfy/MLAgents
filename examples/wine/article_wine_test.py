from datetime import timedelta

import pandas as pd
from io import StringIO
from dotenv import load_dotenv

from utils.print_details import display_explanation_results, display_validation_results, \
    display_feature_engineering_results, display_target_analysis_results, display_data_analysis_results, \
    display_improvement_details, display_solution_tracking_details
from workflow import create_ml_workflow, get_analysis_results, get_validation_results, get_explanation_results, \
    get_improvement_results, get_execution_results
from utils.data_utils import create_data_summary_from_string

load_dotenv()

def main():
    print("=== WINE QUALITY DATASET ANALYSIS WITH SOLUTION TRACKING ===")

    wine_qt = pd.read_csv('winequality-red.csv')
    print("Dataset preview:")
    print(wine_qt.head())
    print(f"Dataset shape: {wine_qt.shape}")
    print()

    workflow = create_ml_workflow()
    csv_string = wine_qt.to_csv(index=False)
    data_summary = create_data_summary_from_string(csv_string)

    initial_state = {
        "data_file_path": "winequality-red.csv",
        "data_summary": data_summary,
        "user_objective": "I want to classify wine quality with accuracy of 85% and 10-fold cross validation",
        "iteration_count": 0,
        "max_iterations": 10,
        "improvement_history": [],
        "messages": [],
        "data_analysis": {},
        "target_analysis": {},
        "feature_engineering": {},
        "solution_history": [],
        "current_best_solution": None,
        "change_focus": None,
        "execution_time": timedelta(0),
        "total_tokens": 0
    }

    print("Running ML workflow analysis with solution tracking...")
    result = workflow.invoke(initial_state, {"recursion_limit": 50})

    analysis = get_analysis_results(result)
    validation = get_validation_results(result)
    explanation = get_explanation_results(result)
    improvement = get_improvement_results(result)

    print("\n=== FINAL ANALYSIS RESULTS ===")

    display_data_analysis_results(analysis["data_analysis"])

    display_target_analysis_results(analysis["target_analysis"])

    display_feature_engineering_results(analysis["feature_engineering"])

    display_validation_results(validation)

    display_explanation_results(explanation)

    display_solution_tracking_details(analysis, result)
    display_improvement_details(improvement)
    print(f"EXECUTION TIME!: {analysis['execution_time']}")
    print(f"TOTAL TOKENS: {analysis['total_tokens']}")
    return result


if __name__ == "__main__":
    main()