from groq import Groq
import re
import json
import pandas as pd
from typing import List, Dict, Any
import dirtyjson
import ast

api_key  ='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'

client = Groq(api_key=api_key)

def call_llm(model,sys,user):
    try:
        response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user}
        ],
        temperature=0
    )
        return response.choices[0].message.content
    except Exception as e:
        print("model inference error", e)
        return ''
    
def try_parse(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        try:
            return (ast.literal_eval(s))
        except Exception as e:
            return None
    
def fix_multiline_strings(s):
    # ONLY fix newlines inside quotes (rough heuristic)
    return re.sub(r'"\s*\n\s*', '"\\n', s)

def extract_json_from_text(text):
    parsed = try_parse(text)
    results = []
    if not parsed:
        pattern = r'```(?:json)?\s*\n?(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)

        

        for match in matches:
            try:
                raw = match.strip()

                parsed = try_parse(raw)
                if not parsed:
                    cleaned = fix_multiline_strings(match.strip())
                    parsed = try_parse(cleaned)
                    if not parsed:
                        dirty_parsed = dirtyjson.loads(cleaned)
                        parsed = dict(dirty_parsed)
                results.append(parsed)
            except json.JSONDecodeError:
                continue  # skip invalid JSON

        if not results:
             return None
    else:
        results.append(parsed)
    return results if len(results) > 1 else results[0]


def prepare_ground_truths(df: pd.DataFrame, res_columns: List[str]) -> List[Dict[str, Any]]:
    """
    Dynamically build ground truth list from specified columns.
    
    Args:
        df: DataFrame containing ground truth data
        res_columns: List of column names to extract as ground truth keys
    
    Returns:
        List of dictionaries with ground truth values
    """
    ground_truths = []
    
    for _, row in df.iterrows():
        gt_dict = {}
        for col in res_columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame. Available columns: {df.columns.tolist()}")
            # Use column name as key, or derive a cleaner key name
            key = col.lower().replace(' ', '_').replace('ground_truth_', '')
            gt_dict[key] = row[col]
        ground_truths.append(gt_dict)
    
    return ground_truths