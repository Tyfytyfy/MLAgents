import json
import subprocess
import tempfile
import os
import time
import traceback
import sys
import hashlib
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from states.data_state import MLState, CodeExecutionMessage

previous_code_hashes = set()

load_dotenv()

def code_executor_agent_node(state: MLState) -> MLState:
    print(state.get('execution_time'))
    print(state.get('total_tokens'))
    global previous_code_hashes
    start = time.time()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = None

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

    if not generated_code:
        error_message = CodeExecutionMessage(
            role="assistant",
            content="Code execution failed: No generated code found",
            type="code_execution",
            execution_status="error",
            model_metrics={},
            execution_logs="",
            error_details="No generated code available for execution",
            saved_model_path="",
            feature_importance={}
        )
        end_time = time.time()
        existing_messages = state.get("messages", [])
        return create_return_state(state, existing_messages + [error_message], response=None, duration=end_time - start)

    current_iteration = state.get("iteration_count", 0)
    code_hash = hashlib.md5(generated_code.encode()).hexdigest()

    print("=== CODE DUPLICATE CHECK ===")
    print(f"Current iteration: {current_iteration}")
    #print(f"Current code hash: {code_hash}")
    #print(f"Previous hashes: {previous_code_hashes}")

    # if code_hash in previous_code_hashes:
    #     print("DUPLICATE CODE DETECTED!")
    #
    #     error_message = CodeExecutionMessage(
    #         role="assistant",
    #         content="Code execution failed: Duplicate code generated",
    #         type="code_execution",
    #         execution_status="error",
    #         model_metrics={},
    #         execution_logs="",
    #         error_details="Code generator produced identical code as previous iteration. This indicates the improvement context is not being properly utilized.",
    #         saved_model_path="",
    #         feature_importance={}
    #     )
    #
    #     existing_messages = state.get("messages", [])
    #     return create_return_state(state, existing_messages + [error_message])
    # else:
    #     print("New code detected - different from previous iterations")
    #
    # previous_code_hashes.add(code_hash)
    # print("=== END DUPLICATE CHECK ===")

    #print("=== GENERATED CODE DEBUG ===")
    #print("First 500 characters of generated code:")
    #print(generated_code[:500])
    #print("=== END CODE DEBUG ===")

    execution_result = execute_python_code_with_auto_install(generated_code, state["data_file_path"])

    if execution_result["status"] == "error":
        analysis_prompt = f"""
        Analyze this code execution error:
        
        Generated Code:
        {generated_code[:1000]}...
        
        Error Details:
        {execution_result['error']}
        
        Execution Logs:
        {execution_result['logs']}
        
        Provide analysis of what went wrong and suggestions for fixes.
        """

        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        error_analysis = response.content

        #print("=== CODE EXECUTOR ERROR ANALYSIS ===")
        #print(error_analysis)
        #print("=== END ERROR ANALYSIS ===")
    else:
        error_analysis = ""

    execution_message = CodeExecutionMessage(
        role="assistant",
        content=f"Code execution {execution_result['status']}: {execution_result.get('summary', '')}",
        type="code_execution",
        execution_status=execution_result["status"],
        model_metrics=execution_result.get("metrics", {}),
        execution_logs=execution_result["logs"],
        error_details=execution_result.get("error", "") + "\n" + error_analysis,
        saved_model_path=execution_result.get("model_path", ""),
        feature_importance=execution_result.get("feature_importance", {})
    )

    #print("=== CODE EXECUTOR AGENT RESULTS ===")
    #print(f"Status: {execution_result['status']}")
    #print(f"Logs: {execution_result['logs']}")
    if execution_result.get('metrics'):
        print(f"Metrics: {execution_result['metrics']}")
    if execution_result.get('feature_importance'):
        print(f"Feature Importance: {execution_result['feature_importance']}")
    #print("=== END CODE EXECUTOR RESULTS ===")

    existing_messages = state.get("messages", [])
    end = time.time()
    return create_return_state(state, existing_messages + [execution_message], response, duration=end - start)


def execute_python_code_with_auto_install(code: str, file_path: str) -> dict:
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"=== EXECUTION DEBUG ===")
            #print(f"Using temp directory: {temp_dir}")
            #print(f"Source data file: {file_path}")

            data_file = os.path.join(temp_dir, "data.csv")
            import shutil
            shutil.copy2(file_path, data_file)

            #print(f"Copied data to: {data_file}")
            #print(f"Data file exists: {os.path.exists(data_file)}")

            data_file_python = data_file.replace("\\", "/")

            import re
            modified_code = re.sub(
                r"pd\.read_csv\s*\(\s*['\"][^'\"]*['\"]",
                f"pd.read_csv('{data_file_python}'",
                code
            )

            if "pd.read_csv(" not in modified_code and "read_csv(" not in modified_code:
                print("WARNING: No pd.read_csv found, adding data loading code")
                data_loading = f"""
import pandas as pd
import numpy as np

data = pd.read_csv('{data_file_python}')

"""
                modified_code = data_loading + modified_code
            else:
                print("Found pd.read_csv in code")

            auto_install_code = f"""
import subprocess
import sys

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {{package}}")
    except Exception as e:
        print(f"Failed to install {{package}}: {{e}}")

try:
    import imblearn
except ImportError:
    print("Installing imbalanced-learn...")
    install_package("imbalanced-learn")
    import imblearn

{modified_code}
"""

            code_file = os.path.join(temp_dir, "ml_pipeline.py")
            with open(code_file, "w") as f:
                f.write(auto_install_code)

            #print(f"Written code file to: {code_file}")
            #print(f"Code file size: {os.path.getsize(code_file)} bytes")
            #print("=== END EXECUTION DEBUG ===")

            result = subprocess.run(
                [sys.executable, code_file],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=600
            )

            if result.returncode == 0:
                output = result.stdout
                metrics = extract_metrics_from_output(output)
                feature_importance = extract_feature_importance(temp_dir, output)

                print("=== EXECUTION SUCCESS DEBUG ===")
                #print(f"Return code: {result.returncode}")
               # print(f"Output length: {len(output)} chars")
                #print(f"Extracted metrics: {metrics}")
                #print(f"Feature importance: {feature_importance}")
                #print("=== END SUCCESS DEBUG ===")

                return {
                    "status": "success",
                    "logs": output,
                    "metrics": metrics,
                    "feature_importance": feature_importance,
                    "summary": "Code executed successfully with auto-install",
                    "model_path": ""
                }
            else:
                #print("=== EXECUTION FAILURE DEBUG ===")
                #print(f"Return code: {result.returncode}")
                #print(f"STDOUT: {result.stdout}")
                #print(f"STDERR: {result.stderr}")
                #print("=== END FAILURE DEBUG ===")

                return {
                    "status": "error",
                    "logs": result.stdout,
                    "error": result.stderr,
                    "summary": "Code execution failed",
                    "feature_importance": {},
                    "model_path": ""
                }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "logs": "",
            "error": "Code execution timed out after 10 minutes",
            "summary": "Execution timeout",
            "feature_importance": {},
            "model_path": ""
        }
    except Exception as e:
        return {
            "status": "error",
            "logs": "",
            "error": f"Execution error: {str(e)}\n{traceback.format_exc()}",
            "summary": "Execution exception",
            "feature_importance": {},
            "model_path": ""
        }


def create_return_state(state: MLState, messages: list, response, duration) -> MLState:
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
        "messages": messages,
        "data_analysis": state.get("data_analysis", {}),
        "target_analysis": state.get("target_analysis", {}),
        "feature_engineering": state.get("feature_engineering", {}),
        "solution_history": state.get("solution_history", []),
        "current_best_solution": state.get("current_best_solution"),
        "change_focus": state.get("change_focus"),
        "execution_time": state.get("execution_time", timedelta(0)) + timedelta(seconds=duration),
        "total_tokens": state.get("total_tokens", 0) + total_tokens
    }


def extract_feature_importance(temp_dir: str, output: str) -> dict:
    feature_importance = {}

    try:
        importance_file = os.path.join(temp_dir, "feature_importance.json")
        if os.path.exists(importance_file):
            with open(importance_file, 'r') as f:
                feature_importance = json.load(f)
            print(f"=== FEATURE IMPORTANCE LOADED FROM FILE ===")
            print(feature_importance)
        else:
            print("=== EXTRACTING FEATURE IMPORTANCE FROM LOGS ===")
            lines = output.split('\n')
            for line in lines:
                line = line.strip()
                if ':' in line and any(keyword in line.lower() for keyword in ['feature', 'importance', 'coefficient']):
                    try:
                        parts = line.split(':')
                        if len(parts) == 2:
                            feature_name = parts[0].strip()
                            importance_value = float(parts[1].strip())
                            feature_importance[feature_name] = importance_value
                    except:
                        continue

    except Exception as e:
        print(f"Error extracting feature importance: {e}")

    return feature_importance


def extract_metrics_from_output(output: str) -> dict:
    metrics = {}
    lines = output.split('\n')

    for line in lines:
        line = line.strip()

        if 'accuracy' in line.lower() and ':' in line:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    value_str = parts[1].strip()
                    value_str = ''.join(c for c in value_str if c.isdigit() or c == '.')
                    if value_str:
                        metrics['accuracy'] = float(value_str)
            except:
                pass

    return metrics