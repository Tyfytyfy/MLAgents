import json
import time
from datetime import timedelta

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agent_utils.data_agent_utils import data_agent_prompt
from states.data_state import MLState, DataAnalysisMessage
from utils.data_utils import get_data_for_llm_analysis, data_len

load_dotenv()


def data_agent_node(state: MLState) -> MLState:
    print(state.get('execution_time'))
    print(state.get('total_tokens'))
    time.sleep(3)
    start = time.time()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    data_for_analysis = get_data_for_llm_analysis(state['data_summary'])
    prompt = data_agent_prompt(state.get('user_objective'), data_for_analysis)

    response = llm.invoke([HumanMessage(content=prompt)])
    result = response.content

    print("=== DATA AGENT RAW RESPONSE ===")
    print(result)
    print("=== END DATA AGENT RESPONSE ===")

    if "```json" in result:
        json_str = result.split("```json")[1].split("```")[0].strip()
    else:
        json_str = result

    parsed_result = json.loads(json_str)

    data_message = DataAnalysisMessage(
        role="assistant",
        content=f"Data analysis completed: {json.dumps(parsed_result)}",
        type="data_analysis",
        data_profile=parsed_result["profile"],
        columns=parsed_result["columns"],
        data_types=parsed_result["types"],
        missing_data=parsed_result["missing"],
        outliers_info=parsed_result["outliers"],
        data_length=data_len(state['data_summary'])
    )

    existing_messages = state.get("messages", [])
    end = time.time()
    return {
        "data_file_path": state["data_file_path"],
        "data_summary": state["data_summary"],
        "user_objective": state["user_objective"],
        "iteration_count": state.get("iteration_count", 0),
        "max_iterations": state.get("max_iterations", 3),
        "improvement_history": state.get("improvement_history", []),
        "messages": existing_messages + [data_message],
        "data_analysis": {
            "profile": parsed_result["profile"],
            "columns": parsed_result["columns"],
            "types": parsed_result["types"],
            "missing": parsed_result["missing"],
            "outliers": parsed_result["outliers"],
            "data_length": data_len(state['data_summary'])
        },
        "target_analysis": state.get("target_analysis", {}),
        "feature_engineering": state.get("feature_engineering", {}),
        "solution_history": state.get("solution_history", []),
        "current_best_solution": state.get("current_best_solution"),
        "change_focus": state.get("change_focus"),
        "execution_time": state.get("execution_time", timedelta(0)) + timedelta(seconds=end - start),
        "total_tokens": state.get("total_tokens", 0) + response.usage_metadata.get('total_tokens', 0)
    }