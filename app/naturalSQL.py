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

# implementing the generate sql query function

def generate_sql_query(user_message:str,schemas:dict,max_attempts:int=1)->dict:
    formatted_system_message = f"""
    {load_system_message(schemas)}

    IMPORTANT: Your response must be valid JSON matching this schema:
    {json.dumps(DECISION_LOG_SCHEMA, indent=2)}

    Ensure all responses strictly follow this format.  Include a final_summary and visualization_suggestion in the decision_log.
    """
    for attempt in range(max_attempts):
        try:
            response = get_completion_from_messages(formatted_system_message, user_message)
            json_response = json.loads(response)    

            if not validate_response_structure(json_response):
                logger.warning(f"Invalid response structure. Attempt: {attempt + 1}")
                continue

            return {
                "query": json_response.get('query'),
                "error": json_response.get('error'),
                "decision_log": json_response['decision_log'],
                "visualization_recommendation": json_response['decision_log'].get('visualization_suggestion')
            }

        except json.JSONDecodeError as e:
            logger.exception(f"Invalid JSON response: {response}, Error: {e}")
            continue
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            continue

    return {
        "error": "Failed to generate a valid SQL query after multiple attempts.",
        "decision_log": {
            "execution_feedback": ["Failed to generate a valid response after multiple attempts."],
            "final_summary": "Query generation failed."
        }
    }

# implement response validation

def validate_response_structure(response:dict)->bool:
    """Validates the structure of the Gemini response against the schema."""
    try:
        if not all(key in response for key in ["query","decision log"]):
            return False
        decision_log = response["decision_log"]
        required_sections = [
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

        if not all(key in decision_log for key in required_sections):
            return False
        for path in decision_log["path_identification"]:
            if not all(key in path for key in ["description", "tables", "columns", "score"]):
                return False

        for explanation in decision_log["chosen_path_explanation"]:
            if not all(key in explanation for key in ["table", "columns", "reason"]):
                return False
        
        return True
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return False
    
def build_markdown_decision_log(decision_log:Dict)->str:
    """
    Builds a markdown formatted decision log that matches the schema structure.
    Handles all fields defined in the DECISION_LOG_SCHEMA.
    """
    markdown_log=[]

    # query input details
    if query_details := decision_log.get("query_input_details"):
        markdown_log.extend([
             "### Query Input Analysis",
            "\n".join(f"- {detail}" for detail in query_details),
            ""
        ])

    #preprocessing steps
    if preprocessing := decision_log.get("preprocessing_steps"):
        markdown_log.extend([
            "### Preprocessing Steps",
            "\n".join(f"- {step}" for step in preprocessing),
            ""
        ])

    # path identification
    if paths := decision_log.get("path_identification"):
        markdown_log.extend([
            "### Path Identification",
            "\n".join([
                f"**Path {i+1}** (Score: {path['score']})\n"
                f"- Description: {path['description']}\n"
                f"- Tables: {', '.join(path['tables'])}\n"
                f"- Columns: {', '.join([', '.join(cols) for cols in path['columns']])}"
                for i, path in enumerate(paths)
            ]),
            ""
        ])

    # ambiguity detection
    if ambiguities := decision_log.get("ambiguity_detection"):
        markdown_log.extend([
            "### Ambiguity Analysis",
            "\n".join(f"- {ambiguity}" for ambiguity in ambiguities),
            ""
        ])

     # resolution criteria
    if criteria := decision_log.get("resolution_criteria"):
        markdown_log.extend([
            "### Resolution Criteria",
            "\n".join(f"- {criterion}" for criterion in criteria),
            ""
        ])

    # chosen path explanation
    if chosen_path := decision_log.get("chosen_path_explanation"):
        markdown_log.extend([
            "### Selected Tables and Columns",
            "\n".join([
                f"**{table['table']}**\n"
                f"- Columns: {', '.join(table['columns'])}\n"
                f"- Reason: {table['reason']}"
                for table in chosen_path
            ]),
            ""
        ])

    # generated sql query
    if sql_query := decision_log.get("generated_sql_query"):
        markdown_log.extend([
            "### Generated SQL Query",
            f"```sql\n{sql_query}\n```",
            ""
        ])

    # alternative paths
    if alternatives := decision_log.get("alternative_paths"):
        markdown_log.extend([
            "### Alternative Approaches",
            "\n".join(f"- {alt}" for alt in alternatives),
            ""
        ])
    
    # execution feedback
    if feedback := decision_log.get("execution_feedback"):
        markdown_log.extend([
            "### Execution Feedback",
            "\n".join(f"- {item}" for item in feedback),
            ""
        ])
    
    # finalis summary
    if summary := decision_log.get("final_summary"):
        markdown_log.extend([
            "### Summary",
            summary,
            ""
        ])
    
    # visualisation suggestions
    if viz_suggestion := decision_log.get("visualization_suggestion"):
        markdown_log.extend([
            "### Visualization Recommendation",
            f"Suggested visualization type: `{viz_suggestion}`",
            ""
        ])
    
    # Join with proper line breaks and clean up any extra spaces
    return "\n".join(line.rstrip() for line in markdown_log)

def create_chart(df:pd.DataFrame,chart_type:str,x_col:col,y_col:str)->Optional[alt.Chart]:
    """Create a chart using Altair library."""
    base_chart = alt.Chart(df).configure_title(fontSize=18,fontWeight='bold',font='Roboto')

    try:
        chart_props={
            "Bar Chart": base_chart.mark_bar(),
            "Line Chart": base_chart.mark_line(),
            "Scatter Plot": base_chart.mark_circle(),
            "Area Chart": base_chart.mark_area(),
            "Histogram": base_chart.mark_bar()
        }

        if chart_type == "Histogram":
            chart = chart_props[chart_type].encode(
                alt.X(x_col, bin=alt.Bin(maxbins=30), title=x_col),
                y=alt.Y('count()', title='Count')
            ).properties(
                width='container',
                height=400
            ).interactive()
        else:
            encoding = {
                "x": alt.X(x_col, title=x_col),
                "y": alt.Y(y_col, title=y_col)
            }
            if chart_type in ["Bar Chart", "Line Chart"]:
                encoding["color"] = alt.Color(y_col, legend=None)
            elif chart_type == "Scatter Plot":
                encoding["tooltip"] = [x_col, y_col]

            chart = chart_props[chart_type].encode(**encoding).properties(
                width='container',
                height=400
            ).interactive()

        return chart

    except Exception as e:
        st.error(f"Error generating the chart: {e}")
        logger.error(f"Error generating chart: {e}")
        return None
    
def display_summary_statistics(df:pd.DataFrame)->None:
    """Display summary statistics for the given DataFrame."""

    if df.empty:
        st.warning("dataframe is empty so not able to display results")
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    tab1, tab2 = st.tabs(["Numeric Summary Statistics", "Categorical Data Insights"])

    if not numeric_cols.empty:
        with tab1:
            numeric_stats = df[numeric_cols].describe().T
            numeric_stats['median'] = df[numeric_cols].median()
            numeric_stats['mode'] = df[numeric_cols].mode().fillna(0).iloc[0]
            numeric_stats['iqr'] = numeric_stats['75%'] - numeric_stats['25%']
            numeric_stats['skew'] = df[numeric_cols].skew()
            numeric_stats['kurt'] = df[numeric_cols].kurt()

            st.markdown("### Numeric Summary Statistics")
            st.dataframe(numeric_stats.style.format("{:.2f}").highlight_max(axis=0, color="lightgreen"))

            for col in numeric_cols:
                st.markdown(f"#### {col}")
                chart = alt.Chart(df).mark_bar().encode(
                    alt.X(col, bin=alt.Bin(maxbins=30), title=f"Distribution of {col}"),
                    y='count()'
                ).properties(
                    width='container',
                    height=200
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

    if not non_numeric_cols.empty:
        with tab2:
            st.markdown("### Categorical Data Insights")
            for col in non_numeric_cols:
                st.markdown(f"**{col} Frequency**")
                freq_table = df[col].value_counts().reset_index()
                freq_table.columns = ['Category', 'Count']
                freq_table['Percentage'] = (freq_table['Count'] / len(df) * 100).round(2)
                st.table(freq_table.style.format({"Percentage": "{:.2f}%"}))




        

    
    







    