import time
from datetime import timedelta
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_utils.feature_agent_utils import no_focus_change_feature_agent_prompt, focus_change_feature_agent_prompt, \
    create_feature_message_from_config, create_feature_return_state
from states.data_state import MLState
from utils.data_utils import get_data_for_llm_analysis
from utils.extract_json import extract_json_from_output

load_dotenv()


def feature_agent_node(state: MLState) -> MLState:
    print(state.get('execution_time'))
    print(state.get('total_tokens'))
    start = time.time()

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    data_analysis = state.get("data_analysis", {})
    target_analysis = state.get("target_analysis", {})
    data_for_analysis = get_data_for_llm_analysis(state['data_summary'])

    improvement_context = get_improvement_context(state)
    current_iteration = state.get("iteration_count", 0)

    change_focus = state.get("change_focus", None)
    best_solution = state.get("current_best_solution", None)

    if change_focus is None or best_solution is None:
        prompt = no_focus_change_feature_agent_prompt(state.get("user_objective", None), data_analysis,
                                                      target_analysis, data_for_analysis)
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content
        parsed_result = extract_json_from_output(result)
        message = create_feature_message_from_config(parsed_result, "Initial feature engineering config")
        end = time.time()
        return create_feature_return_state(state, message, response, end - start)
    elif change_focus == 'encoding_method':
        previous_config = best_solution.get('feature_engineering', None)
        if previous_config is None:
            prompt = no_focus_change_feature_agent_prompt(state.get("user_objective", ""), data_analysis,
                                                          target_analysis, data_for_analysis)
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content
            parsed_result = extract_json_from_output(result)
            message = create_feature_message_from_config(parsed_result, "Initial feature engineering config")
            end = time.time()
            return create_feature_return_state(state, message, response, end - start)

        encoding_method = previous_config.get("encoding_method")
        encoding_method_prompt = focus_change_feature_agent_prompt(state.get("user_objective", ""), data_analysis,
                                                                   target_analysis, data_for_analysis, encoding_method,
                                                                   change_focus)
        response = llm.invoke([HumanMessage(content=encoding_method_prompt)])
        result = response.content
        parsed_result = extract_json_from_output(result)
        updated_config = previous_config.copy()
        updated_config['encoding_method'] = parsed_result['encoding_method']
        message = create_feature_message_from_config(updated_config, "Updated feature config with new encoding method")
        end = time.time()
        return create_feature_return_state(state, message, response, end - start)
    elif change_focus == 'scaling_method':
        previous_config = best_solution.get('feature_engineering', None)
        if previous_config is None:
            prompt = no_focus_change_feature_agent_prompt(state.get("user_objective", ""), data_analysis,
                                                          target_analysis, data_for_analysis)
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content
            parsed_result = extract_json_from_output(result)
            message = create_feature_message_from_config(parsed_result, "Initial feature engineering config")
            end = time.time()
            return create_feature_return_state(state, message, response, end - start)

        scaling_method = previous_config.get("scaling_method")
        scaling_method_prompt = focus_change_feature_agent_prompt(state.get("user_objective", ""), data_analysis,
                                                                  target_analysis, data_for_analysis, scaling_method,
                                                                  change_focus)
        response = llm.invoke([HumanMessage(content=scaling_method_prompt)])
        result = response.content
        parsed_result = extract_json_from_output(result)
        updated_config = previous_config.copy()
        updated_config['scaling_method'] = parsed_result['scaling_method']
        message = create_feature_message_from_config(updated_config, "Updated feature config with new scaling method")
        end = time.time()
        return create_feature_return_state(state, message, response, end - start)
    elif change_focus == 'feature_selection':
        previous_config = best_solution.get('feature_engineering', None)
        if previous_config is None:
            prompt = no_focus_change_feature_agent_prompt(state.get("user_objective", ""), data_analysis,
                                                          target_analysis, data_for_analysis)
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content
            parsed_result = extract_json_from_output(result)
            message = create_feature_message_from_config(parsed_result, "Initial feature engineering config")
            end = time.time()
            return create_feature_return_state(state, message, response, end - start)

        feature_selection = previous_config.get("features_to_drop")
        feature_selection_prompt = focus_change_feature_agent_prompt(state.get("user_objective", ""), data_analysis,
                                                                     target_analysis, data_for_analysis,
                                                                     feature_selection,
                                                                     change_focus)
        response = llm.invoke([HumanMessage(content=feature_selection_prompt)])
        result = response.content
        parsed_result = extract_json_from_output(result)
        updated_config = previous_config.copy()
        updated_config['features_to_drop'] = parsed_result['features_to_drop']
        message = create_feature_message_from_config(updated_config,
                                                     "Updated feature config with new feature selection")
        end = time.time()
        return create_feature_return_state(state, message, response, end - start)
    elif change_focus == 'class_balancing':
        previous_config = best_solution.get('feature_engineering', None)
        if previous_config is None:
            prompt = no_focus_change_feature_agent_prompt(state.get("user_objective", ""), data_analysis,
                                                          target_analysis, data_for_analysis)
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content
            parsed_result = extract_json_from_output(result)
            message = create_feature_message_from_config(parsed_result, "Initial feature engineering config")
            end = time.time()
            return create_feature_return_state(state, message, response, end - start)

        class_balancing = previous_config.get("preprocessing_steps")
        class_balancing_prompt = focus_change_feature_agent_prompt(state.get("user_objective", ""), data_analysis,
                                                                   target_analysis, data_for_analysis, class_balancing,
                                                                   change_focus)
        response = llm.invoke([HumanMessage(content=class_balancing_prompt)])
        result = response.content
        parsed_result = extract_json_from_output(result)
        updated_config = previous_config.copy()
        updated_config['preprocessing_steps'] = parsed_result['preprocessing_steps']
        message = create_feature_message_from_config(updated_config,
                                                     "Updated feature config with new preprocessing steps")
        end = time.time()
        return create_feature_return_state(state, message, response, end - start)
    else:
        if best_solution is not None:
            previous_config = best_solution.get('feature_engineering', None)
            if previous_config is not None:
                message = create_feature_message_from_config(previous_config,
                                                             "Copied previous feature engineering config")
                end = time.time()
                return create_feature_return_state(state, message, None, end - start)

        prompt = no_focus_change_feature_agent_prompt(state.get("user_objective", ""), data_analysis,
                                                      target_analysis, data_for_analysis)
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content
        parsed_result = extract_json_from_output(result)
        message = create_feature_message_from_config(parsed_result, "Initial feature engineering config")
        end = time.time()
        return create_feature_return_state(state, message, response, end - start)


def get_improvement_context(state: MLState) -> str:
    messages = state.get("messages", [])
    current_iteration = state.get("iteration_count", 0)

    if current_iteration == 0:
        return ""

    context = f"PREVIOUS FAILURE ANALYSIS - ITERATION {current_iteration}:\n\n"

    execution_results = []
    improvement_actions = []
    validation_results = []
    error_details = []

    for message in reversed(messages):
        msg_data = None

        if hasattr(message, 'additional_kwargs'):
            msg_data = message.additional_kwargs
        elif isinstance(message, dict):
            msg_data = message

        if not msg_data:
            continue

        msg_type = msg_data.get("type")

        if msg_type == "code_execution":
            metrics = msg_data.get("model_metrics", {})
            error = msg_data.get("error_details", "")
            status = msg_data.get("execution_status", "")

            if metrics:
                execution_results.append(f"Status: {status}, Metrics: {metrics}")
            if error:
                error_details.append(error)

        elif msg_type == "validation":
            assessment = msg_data.get("model_assessment", "")
            suggestions = msg_data.get("improvement_suggestions", [])

            if assessment:
                validation_results.append(assessment)
            if suggestions:
                improvement_actions.extend(suggestions)

        elif msg_type == "improvement":
            actions = msg_data.get("recommended_actions", [])
            root_cause = msg_data.get("root_cause_analysis", "")

            if root_cause:
                validation_results.append(root_cause)
            if actions:
                improvement_actions.extend(actions)

    if execution_results:
        context += f"PREVIOUS EXECUTION RESULTS:\n{execution_results[0]}\n\n"

    if error_details:
        context += f"ERROR DETAILS:\n{error_details[0]}\n\n"

    if validation_results:
        context += f"FAILURE ANALYSIS:\n{validation_results[0]}\n\n"

    if improvement_actions:
        context += "REQUIRED CHANGES:\n"
        for i, action in enumerate(improvement_actions[:5], 1):
            context += f"{i}. {action}\n"
        context += "\n"

    context += "CRITICAL: You must use COMPLETELY DIFFERENT approach than what failed before!\n"

    return context