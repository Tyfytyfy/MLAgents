import pandas as pd
import random
from typing import Dict, Any


def create_data_summary(file_path: str, sample_size: int = 1000) -> Dict[str, Any]:
    try:
        df = pd.read_csv(file_path)

        shape = df.shape
        columns = df.columns.tolist()
        dtypes = df.dtypes.astype(str).to_dict()

        sample_df = df.head(min(sample_size, len(df)))
        sample_csv = sample_df.to_csv(index=False)

        numeric_cols = df.select_dtypes(include=['number']).columns
        stats = {}
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().to_dict()

        missing_info = df.isnull().sum().to_dict()
        missing_info = {k: v for k, v in missing_info.items() if v > 0}

        categorical_info = {}
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 20:
                categorical_info[col] = df[col].value_counts().head(10).to_dict()
            else:
                categorical_info[col] = {"unique_count": unique_count, "sample_values": df[col].unique()[:10].tolist()}

        return {
            "file_path": file_path,
            "shape": shape,
            "columns": columns,
            "dtypes": dtypes,
            "sample_csv": sample_csv,
            "statistics": stats,
            "missing_values": missing_info,
            "categorical_info": categorical_info,
            "sample_size_used": len(sample_df)
        }

    except Exception as e:
        return {
            "file_path": file_path,
            "error": f"Failed to process file: {str(e)}",
            "shape": [0, 0],
            "columns": [],
            "dtypes": {},
            "sample_csv": "",
            "statistics": {},
            "missing_values": {},
            "categorical_info": {},
            "sample_size_used": 0
        }


def get_data_for_llm_analysis(data_summary: dict) -> str:
#Sample Data (first {data_summary['sample_size_used']} rows):
#{data_summary['sample_csv']}
    if data_summary.get("error"):
        return f"Error loading data: {data_summary['error']}"

    analysis_text = f"""
Dataset Summary:
- File: {data_summary['file_path']}
- Shape: {data_summary['shape'][0]} rows, {data_summary['shape'][1]} columns
- Columns: {data_summary['columns']}
- Data Types: {data_summary['dtypes']}



Statistical Summary:
{data_summary['statistics']}

Missing Values:
{data_summary['missing_values'] if data_summary['missing_values'] else 'No missing values'}

Categorical Information:
{data_summary['categorical_info']}

Outlier information:
{data_summary['outlier_info']}
"""
    return analysis_text


def create_data_summary_from_string(data_string: str) -> Dict[str, Any]:

    try:
        from io import StringIO
        df = pd.read_csv(StringIO(data_string))

        shape = df.shape
        columns = df.columns.tolist()
        dtypes = df.dtypes.astype(str).to_dict()

        sample_csv = df.head(min(10, len(df))).to_csv(index=False)

        numeric_cols = df.select_dtypes(include=['number']).columns
        stats = {}
        if len(numeric_cols) > 0:
            stats = df[numeric_cols].describe().to_dict()

        outlier_info = {}
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers_count = outliers_mask.sum()
                outlier_percentage = (outliers_count / len(df)) * 100

                if outliers_count > 0:
                    outlier_info[col] = {
                        "count": int(outliers_count),
                        "percentage": round(outlier_percentage, 2),
                        "bounds": {"lower": lower_bound, "upper": upper_bound}
                    }

        missing_info = df.isnull().sum().to_dict()
        missing_info = {k: v for k, v in missing_info.items() if v > 0}

        categorical_info = {}
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_count = df[col].nunique()
            if unique_count <= 20:
                categorical_info[col] = df[col].value_counts().head(10).to_dict()
            else:
                categorical_info[col] = {"unique_count": unique_count, "sample_values": df[col].unique()[:10].tolist()}

        return {
            "file_path": "from_string",
            "shape": shape,
            "columns": columns,
            "dtypes": dtypes,
            #"sample_csv": sample_csv,
            "statistics": stats,
            "missing_values": missing_info,
            "categorical_info": categorical_info,
            "outlier_info": outlier_info,
            #"sample_size_used": len(df.head(1000)),
        }

    except Exception as e:
        return {
            "file_path": "from_string",
            "error": f"Failed to process data string: {str(e)}",
            "shape": [0, 0],
            "columns": [],
            "dtypes": {},
            "sample_csv": "",
            "statistics": {},
            "missing_values": {},
            "categorical_info": {},
            "outlier_info": {},
            "sample_size_used": 0,
        }

def data_len(data_summary: dict):
    return data_summary['shape'][0]