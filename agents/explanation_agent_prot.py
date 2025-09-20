import time

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_utils.explanation_agent_utils import create_explanation_return_state, create_explanation_agent_prompt, \
    create_passed_explanation_state
from states.data_state import MLState, ExplanationMessage
from utils.extract_json import extract_json_from_output
from utils.extract_text import extract_model_type_from_code, get_generated_code_from_messages, \
    get_validation_results_from_messages, get_execution_results_from_messages


def explanation_agent_node(state: MLState) -> MLState:
    print(state.get('execution_time'))
    print(state.get('total_tokens'))
    time.sleep(3)
    start = time.time()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    validation_results = get_validation_results_from_messages(state.get("messages", []))
    execution_results = get_execution_results_from_messages(state.get("messages", []))

    if validation_results.get("validation_status") != "passed":
        end = time.time()
        return create_passed_explanation_state(state, response=None, duration=end-start)

    real_feature_importance = execution_results.get('feature_importance', {})
    generated_code = get_generated_code_from_messages(state.get("messages", []))
    actual_model_type = extract_model_type_from_code(generated_code.get('generated_code', ''))

    prompt = create_explanation_agent_prompt(state, actual_model_type, validation_results, execution_results, real_feature_importance)

    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content

    parsed_result = extract_json_from_output(result)

    explanation_message = ExplanationMessage(
        role="assistant",
        content=f"Model explanation completed: {parsed_result['model_explanation'][:100]}...",
        type="explanation",
        feature_importance=parsed_result["feature_importance"],
        model_explanation=parsed_result["model_explanation"],
        domain_validation=parsed_result["domain_validation"]
    )

    print("=== EXPLANATION AGENT RESULTS ===")
    print(f"Feature Importance: {parsed_result['feature_importance']}")
    print(f"Model Explanation: {parsed_result['model_explanation']}")
    print(f"Domain Validation: {parsed_result['domain_validation']}")
    print("=== END EXPLANATION RESULTS ===")
    end = time.time()
    return create_explanation_return_state(state, explanation_message, response, duration=end-start)
