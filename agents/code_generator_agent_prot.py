import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_utils.code_generator_utils import create_no_change_focus_generator_prompt, create_code_generation_message, \
    create_code_generation_return_state, extract_section, replace_section, \
    change_hyperparameter_tuning, change_algorithm, change_imports
from states.data_state import MLState
from utils.extract_json import extract_json_from_output
from utils.extract_text import extract_algorithm_from_code

load_dotenv()


def code_generator_agent_node(state: MLState) -> MLState:
    print(state.get('execution_time'))
    print(state.get('total_tokens'))
    start = time.time()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    data_analysis = state.get("data_analysis", {})
    target_analysis = state.get("target_analysis", {})
    feature_engineering = state.get("feature_engineering", {})

    change_focus = state.get("change_focus", None)
    best_solution = state.get("current_best_solution", None)

    if change_focus is None:
        prompt = create_no_change_focus_generator_prompt(state.get("user_objective", ""), data_analysis,
                                                         target_analysis, feature_engineering)
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content

        config = extract_json_from_output(result)
        code = config.get("generated_code", "NO CODE FOUND")
        print(code)
        model_section = extract_section(code, 'HYPERPARAMETERS')
        print(f"MODEL DEFINITION SECTION {model_section}")
        scaling_section = extract_section(code, 'SCALING')
        print(f"SCALING SECTION {scaling_section}")
        new_model = "model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)"
        modified_code = replace_section(code, "HYPERPARAMETERS", new_model)
        config["generated_code"] = modified_code
        message = create_code_generation_message(config, "Initial generated code for ML pipeline")
        print("+" * 50)
        print(f'CODE AFTER SECTION CHANGE {config.get("generated_code", "NO CODE FOUND")}')
        model_section = extract_section(config.get("generated_code", "NO CODE FOUND"), 'HYPERPARAMETERS')
        print(f"MODEL DEFINITION SECTION {model_section}")
        print(f"MODEL NAME: {change_hyperparameter_tuning(model_section)}")
        print("+" * 50)
        end = time.time()
        return create_code_generation_return_state(state, message, response, (end-start))

    elif change_focus == "HYPERPARAMETERS":
        if not best_solution or 'generated_code' not in best_solution:
            prompt = create_no_change_focus_generator_prompt(state.get("user_objective", ""), data_analysis,
                                                             target_analysis, feature_engineering)
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content
            config = extract_json_from_output(result)
            message = create_code_generation_message(config, "Generated code with hyperparameter tuning")
            end = time.time()
            return create_code_generation_return_state(state, message, None, (end-start))
        else:
            code = best_solution.get("generated_code", "")
            model_section = extract_section(best_solution.get("generated_code", "NO CODE FOUND"), 'HYPERPARAMETERS')
            new_hyperparameters = change_hyperparameter_tuning(model_section)
            new_section = replace_section(code, "HYPERPARAMETERS", new_hyperparameters)
            config = {
                "generated_code": new_section,
                "pipeline_code": extract_section(code, "FEATURE_ENGINEERING") or "",
                "model_code": extract_section(code, "MODEL_TRAINING") or ""
            }
            message = create_code_generation_message(config, "Modified hyperparameters")
            print(model_section)
            print(f'new section {new_hyperparameters}')
            end = time.time()
            return create_code_generation_return_state(state, message, None, (end - start))

    elif change_focus == "ALGORITHM_SELECTION":
        if not best_solution or 'generated_code' not in best_solution:
            prompt = create_no_change_focus_generator_prompt(state.get("user_objective", ""), data_analysis,
                                                             target_analysis, feature_engineering)
            response = llm.invoke([HumanMessage(content=prompt)])
            result = response.content
            config = extract_json_from_output(result)
            message = create_code_generation_message(config, "Generated code with algorithm selection")
            end = time.time()
            return create_code_generation_return_state(state, message, response, (end - start))
        else:
            original_code = best_solution.get("generated_code", "")

            new_model_line = change_algorithm(original_code)
            new_model_name = extract_algorithm_from_code(new_model_line)

            new_import_line = change_imports(new_model_name)

            modified_code = replace_section(original_code, "HYPERPARAMETERS", new_model_line)
            modified_code = replace_section(modified_code, "ALGORITHM_SELECTION", new_import_line)
            config = {
                "generated_code": modified_code,
                "pipeline_code": extract_section(modified_code, "FEATURE_ENGINEERING") or "",
                "model_code": extract_section(modified_code, "MODEL_TRAINING") or ""
            }

            message = create_code_generation_message(config, "Modified algorithm")
            print(f"New model: {new_model_line}")
            print(f"New import: {new_import_line}")
            end = time.time()
            return create_code_generation_return_state(state, message, None, (end - start))

    else:
        prompt = create_no_change_focus_generator_prompt(state.get("user_objective", ""), data_analysis,
                                                         target_analysis,
                                                         feature_engineering)
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content

        config = extract_json_from_output(result)
        message = create_code_generation_message(config, "Generated code for ML pipeline")
        end = time.time()
        return create_code_generation_return_state(state, message, response, (end - start))

def get_improvement_context(state: MLState) -> str:
    messages = state.get("messages", [])
    current_iteration = state.get("iteration_count", 0)

    if current_iteration == 0:
        return ""

    context_parts = []

    validation_messages = []
    improvement_messages = []
    execution_messages = []

    for message in reversed(messages):
        msg_data = None

        if hasattr(message, 'additional_kwargs'):
            msg_data = message.additional_kwargs
        elif isinstance(message, dict):
            msg_data = message

        if not msg_data:
            continue

        msg_type = msg_data.get("type")

        if msg_type == "validation":
            validation_messages.append(msg_data)
        elif msg_type == "improvement":
            improvement_messages.append(msg_data)
        elif msg_type == "code_execution":
            execution_messages.append(msg_data)

    if validation_messages:
        assessment = validation_messages[0].get("model_assessment", "")
        if assessment:
            context_parts.append(f"ANALYSIS OF PREVIOUS FAILURE CAUSE:\n{assessment}\n")

    if improvement_messages:
        actions = improvement_messages[0].get("recommended_actions", [])
        root_cause = improvement_messages[0].get("root_cause_analysis", "")

        if root_cause:
            context_parts.append(f"ROOT CAUSE ANALYSIS:\n{root_cause}\n")

        if actions:
            context_parts.append("RECOMMENDATIONS FROM THE IMPROVEMENT AGENT:")
            for i, action in enumerate(actions, 1):
                context_parts.append(f"{i}. {action}")
            context_parts.append("")

    if execution_messages:
        error_details = execution_messages[0].get("error_details", "")
        execution_logs = execution_messages[0].get("execution_logs", "")

        if error_details:
            context_parts.append(f"DETAILED ERROR LOG FROM PREVIOUS ITERATION:\n{error_details}\n")
        elif execution_logs:
            context_parts.append(f"EXECUTION LOGS FROM PREVIOUS ITERATION:\n{execution_logs}\n")

    if not context_parts:
        return ""

    context_parts.append(
        "CRITICAL: The new feature engineering plan was formulated based on the ABOVE analysis. Implement it to fix the identified issues.")

    return "\n".join(context_parts)