import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import io
import json
import re
import logging
from typing import Dict,List,Optional,Union,TypedDict
import pandas as pd
import altair as alt
import streamlit as st
import streamlit_nested_layout
from dotenv import load_dotenv
from streamlit_extras.colored_header import colored_header
import numpy as np
from streamlit_extras.chart_container import chart_container
from streamlit_extras.dataframe_explorer import dataframe_explorer
import src.database.DB_Config as DB_Config
from src.prompts.Base_Prompt import SYSTEM_MESSAGE
from src.api.LLM_Config import get_completion_from_messages

logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

SUPPORTED_CHART_TYPES = {
    "Bar Chart": "A chart that presents categorical data with rectangular bars.",
    "Line Chart": "A chart that displays information as a series of data points called 'markers' connected by straight line segments.",
    "Scatter Plot": "A plot that displays values for typically two variables for a set of data.",
    "Area Chart": "A chart that displays quantitative data visually, using the area below the line.",
    "Histogram": "A graphical representation of the distribution of numerical data."
}

st.set_page_config(
    page_icon="🗃️",
    page_title="Transforming Questions into Queries",
    layout="wide"
)

load_dotenv()

@st.cache_resource
def load_system_message(schemas: dict) -> str:
    """Loads and formats the system message with database schemas."""
    return SYSTEM_MESSAGE.format(schemas=json.dumps(schemas, indent=2))

def get_data(query:str,db_name:str,db_type:str,host:Optional[str]=None,user:Optional[str]=None,password:Optional[str]=None)->pd.DataFrame:
    return DB_Config.query_database(query,db_name,db_type,host,user,password)

def save_temp_file(uploaded_file)->str:
    """Saves an uploaded file to a temporary location."""
    temp_file_path="temp_database.db"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path

# defining classes
class Path(TypedDict):
    """
    Represents a data path in a data processing or query execution scenario.
    A path can be a series of tables and columns used to answer a specific query or make decisions.

    Attributes:
        description (str): A brief explanation of what this path represents.
        tables (List[str]): A list of table names involved in this path.
        columns (List[List[str]]): A list of lists where each inner list represents columns involved in this path.
        score (int): An integer score that indicates the suitability or preference of this path. Higher values indicate better paths.
    """
    description: str
    tables: List[str]
    columns: List[List[str]]
    score: int

class TableColumn(TypedDict):
    """
    Represents the relationship between a specific table and the columns used in a data processing or query scenario.
    
    Attributes:
        table (str): The name of the table being used.
        columns (List[str]): A list of column names from the table.
        reason (str): A description explaining why these columns are being used (e.g., they are needed for a query or analysis).
    """
    table: str
    columns: List[str]
    reason: str

from typing import Optional, List

class DecisionLog(TypedDict):
    """
    Represents a log of decisions made throughout a data processing or query generation process.
    It includes details such as the input data, steps taken, paths considered, and the generated query.

    Attributes:
        query_input_details (List[str]): A list of details about the query input (e.g., parameters, user input).
        preprocessing_steps (List[str]): A list of preprocessing steps that were applied before generating the SQL query.
        path_identification (List[Path]): A list of identified paths (of type Path) that were considered during the process.
        ambiguity_detection (List[str]): A list of identified ambiguities that might have affected the query generation.
        resolution_criteria (List[str]): A list of criteria used to resolve ambiguities in the decision process.
        chosen_path_explanation (List[TableColumn]): A list of explanations (of type TableColumn) detailing the chosen path's tables and columns.
        generated_sql_query (str): The final SQL query generated after decision-making.
        alternative_paths (List[str]): A list of alternative paths considered, but not chosen.
        execution_feedback (List[str]): Feedback or results obtained after the SQL query was executed.
        final_summary (str): A final summary of the entire decision-making and query generation process.
        visualization_suggestion (Optional[str]): An optional suggestion for a type of data visualization, if relevant.
    """
    query_input_details: List[str]
    preprocessing_steps: List[str]
    path_identification: List[Path]
    ambiguity_detection: List[str]
    resolution_criteria: List[str]
    chosen_path_explanation: List[TableColumn]
    generated_sql_query: str
    alternative_paths: List[str]
    execution_feedback: List[str]
    final_summary: str
    visualization_suggestion: Optional[str]



#defining expected structure of decision log
    
DECISION_LOG_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The generated SQL query"},
        "error": {"type": ["string", "null"], "description": "Error message if query generation failed"},
        "decision_log": {
            "type": "object",
            "properties": {
                "query_input_details": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Details about the input query"
                },
                "preprocessing_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps taken to preprocess the query"
                },
                "path_identification": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "tables": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "columns": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                }
                            },
                            "score": {"type": "integer"}
                        },
                        "required": ["description", "tables", "columns", "score"]
                    }
                },
                "ambiguity_detection": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "resolution_criteria": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "chosen_path_explanation": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "reason": {"type": "string"}
                        },
                        "required": ["table", "columns", "reason"]
                    }
                },
                "generated_sql_query": {"type": "string"},
                "alternative_paths": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "execution_feedback": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "final_summary": {"type": "string"},
                "visualization_suggestion": {"type": ["string", "null"]}
            },
            "required": [
                "query_input_details",
                "preprocessing_steps",
                "path_identification",
                "ambiguity_detection",
                "resolution_criteria",
                "chosen_path_explanation",
                "generated_sql_query",
                "alternative_paths",
                "execution_feedback",
                "final_summary"
            ]
        }
    },
    "required": ["query", "decision_log"]
}




    