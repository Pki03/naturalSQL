import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
import io
import json
import re
import logging
from typing import Dict, List, Optional, Union, TypedDict
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
from typing import Optional, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_CHART_TYPES = {
    "Bar Chart": "A chart that presents categorical data with rectangular bars.",
    "Line Chart": "A chart that displays information as a series of data points called 'markers' connected by straight line segments.",
    "Scatter Plot": "A plot that displays values for typically two variables for a set of data.",
    "Area Chart": "A chart that displays quantitative data visually, using the area below the line.",
    "Histogram": "A graphical representation of the distribution of numerical data."
}

# some basic streamlit ui/ux improvements

st.set_page_config(
    page_icon="🧙‍♂️",
    page_title="NaturalSQL: Transforming Questions into Insights",
    layout="centered",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
        .header {
            text-align: center;
            font-size: 48px;
            font-weight: 600;
            color: #3498db;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin-top: 50px;  /* Margin from the top */
            margin-bottom: 20px;  /* Margin at the bottom */
            padding: 10px;  /* Padding inside the header */
            text-transform: capitalize;
            letter-spacing: 1px;
            border-bottom: 2px solid #3498db;  /* Optional: adds a bottom border for separation */
        }
        .magic {
            animation: smooth-glow 2s ease-in-out infinite;
        }
        @keyframes smooth-glow {
            0% { text-shadow: 0 0 8px #3498db, 0 0 20px #3498db; }
            50% { text-shadow: 0 0 18px #3498db, 0 0 30px #3498db; }
            100% { text-shadow: 0 0 8px #3498db, 0 0 20px #3498db; }
        }
        .header:hover {
            color: #2ecc71;  /* Soft Green on hover */
            cursor: pointer;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="header magic">🧙‍♂️ NaturalSQL</div>', unsafe_allow_html=True)

#loading the api key

load_dotenv()

@st.cache_resource
def load_system_message(_schemas: dict) -> str:
    """Loads and formats the system message with database schemas."""
    return SYSTEM_MESSAGE.format(schemas=json.dumps(_schemas, default=str, indent=2))



def get_data(query: str, db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None) -> pd.DataFrame:
    return DB_Config.query_database(query, db_name, db_type, host, user, password)

def save_temp_file(uploaded_file) -> str:
    """Saves an uploaded file to a temporary location."""
    temp_file_path = "temp_database.db"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    return temp_file_path

# defining classes
class Path(TypedDict):
    description: str
    tables: List[str]
    columns: List[List[str]]
    score: int

class TableColumn(TypedDict):
    table: str
    columns: List[str]
    reason: str

class DecisionLog(TypedDict):
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

# defining expected structure of decision log
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
def generate_sql_query(user_message: str, schemas: dict, db_name: str, db_type: str, host: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None, max_attempts: int = 1) -> dict:
    formatted_system_message = f"""
    {load_system_message(schemas)}

    IMPORTANT: Your response must be valid JSON matching this schema:
    {json.dumps(DECISION_LOG_SCHEMA, indent=2)}

    Ensure all responses strictly follow this format. Include a final_summary and visualization_suggestion in the decision_log.
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
def validate_response_structure(response: dict) -> bool:
    """Validates the structure of the Gemini response against the schema."""
    try:
        if not all(key in response for key in ["query", "decision_log"]):
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

# making the markdown decision log using the defined parameters for decision_log
def build_markdown_decision_log(decision_log: Dict) -> str:
    """
    Builds a markdown formatted decision log that matches the schema structure.
    Handles all fields defined in the DECISION_LOG_SCHEMA, and ensures compatibility with all database types.
    """
    markdown_log = []

    # query input details
    if query_details := decision_log.get("query_input_details"):
        markdown_log.extend([
            "### 🔍 Query Input Analysis",
            "\n".join(f"- {detail}" for detail in query_details),
            ""
        ])

    # preprocessing steps
    if preprocessing := decision_log.get("preprocessing_steps"):
        markdown_log.extend([
            "### 🛠️ Preprocessing Steps",
            "\n".join(f"- {step}" for step in preprocessing),
            ""
        ])

    # path identification
    if paths := decision_log.get("path_identification"):
        markdown_log.extend([
            "### 🚶‍♂️ Path Identification",
            "\n".join([
                f"**Path {i + 1}** (Score: {path['score']})\n"
                f"- ✍️ Description: {path['description']}\n"
                f"- 📚 Tables: {', '.join(path['tables'])}\n"
                f"- 🔑 Columns: {', '.join([', '.join(cols) for cols in path['columns']])}"
                for i, path in enumerate(paths)
            ]),
            ""
        ])

    # ambiguity detection
    if ambiguities := decision_log.get("ambiguity_detection"):
        markdown_log.extend([
            "### ⚖️ Ambiguity Analysis",
            "\n".join(f"- {ambiguity}" for ambiguity in ambiguities),
            ""
        ])

    # resolution criteria
    if criteria := decision_log.get("resolution_criteria"):
        markdown_log.extend([
            "### 📝 Resolution Criteria",
            "\n".join(f"- {criterion}" for criterion in criteria),
            ""
        ])

    # chosen path explanation
    if chosen_path := decision_log.get("chosen_path_explanation"):
        markdown_log.extend([
            "### ✅ Selected Tables and Columns",
            "\n".join([
                f"**{table['table']}**\n"
                f"- 🔑 Columns: {', '.join(table['columns'])}\n"
                f"- 📝 Reason: {table['reason']}"
                for table in chosen_path
            ]),
            ""
        ])

    # generated sql query
    if sql_query := decision_log.get("generated_sql_query"):
        markdown_log.extend([
            "### 🧑‍💻 Generated SQL Query",
            f"```sql\n{sql_query}\n```",
            ""
        ])

    # alternative paths
    if alternatives := decision_log.get("alternative_paths"):
        markdown_log.extend([
            "### 🌱 Alternative Approaches",
            "\n".join(f"- {alt}" for alt in alternatives),
            ""
        ])
    
    # execution feedback
    if feedback := decision_log.get("execution_feedback"):
        markdown_log.extend([
            "### 📊 Execution Feedback",
            "\n".join(f"- {item}" for item in feedback),
            ""
        ])
    
    # final summary
    if summary := decision_log.get("final_summary"):
        markdown_log.extend([
            "### 📋 Summary",
            summary,
            ""
        ])
    
    # visualization suggestions
    if viz_suggestion := decision_log.get("visualization_suggestion"):
        markdown_log.extend([
            "### 📊 Visualization Recommendation",
            f"Suggested visualization type: `{viz_suggestion}`",
            ""
        ])
    
    # Join with proper line breaks and clean up any extra spaces
    return "\n".join(line.rstrip() for line in markdown_log)

def create_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> Optional[alt.Chart]:
    """Create a chart using Altair library."""

    # Create base chart with custom title font style
    base_chart = alt.Chart(df).configure_title(fontSize=18, fontWeight='bold', font='Roboto')

    try:
        # Predefine available chart types mapped to Altair mark types
        chart_props = {
            "Bar Chart": base_chart.mark_bar(),
            "Line Chart": base_chart.mark_line(),
            "Scatter Plot": base_chart.mark_circle(),
            "Area Chart": base_chart.mark_area(),
            "Histogram": base_chart.mark_bar()   # Special handling for binning later
        }

        # Histogram = special case → use binning on X-axis + count frequencies
        if chart_type == "Histogram":
            chart = chart_props[chart_type].encode(
                alt.X(x_col, bin=alt.Bin(maxbins=30), title=x_col),
                y=alt.Y('count()', title='Count')
            ).properties(width='container', height=400).interactive()
        else:
            # General encoding for other chart types
            encoding = {
                "x": alt.X(x_col, title=x_col),
                "y": alt.Y(y_col, title=y_col)
            }

            # Add additional encodings based on chart type
            if chart_type in ["Bar Chart", "Line Chart"]:
                # Coloring by Y values for distinction
                encoding["color"] = alt.Color(y_col, legend=None)
            elif chart_type == "Scatter Plot":
                # Add tooltips for better interactivity
                encoding["tooltip"] = [x_col, y_col]

            # Build final chart
            chart = chart_props[chart_type].encode(**encoding).properties(
                width='container',
                height=400
            ).interactive()

        return chart

    except Exception as e:
        # Error handling: show in Streamlit + log
        st.error(f"Error generating the chart: {e}")
        logger.error(f"Error generating chart: {e}")
        return None

def display_summary_statistics(df: pd.DataFrame) -> None:
    """Display summary statistics for the given DataFrame."""

    if df.empty:
        # Prevent crashing on empty DF
        st.warning("🚨 The dataframe is empty, so we cannot display any results.")
        return
    
    # Separate numeric and non-numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    # Use tabs to neatly separate numeric vs categorical insights
    tab1, tab2 = st.tabs(["📊 Numeric Summary Statistics", "📋 Categorical Data Insights"])

    # ---- NUMERIC COLUMNS ----
    if not numeric_cols.empty:
        with tab1:
            # Compute descriptive statistics
            numeric_stats = df[numeric_cols].describe().T
            numeric_stats['median'] = df[numeric_cols].median()   # Why: median = robust measure of central tendency
            numeric_stats['mode'] = df[numeric_cols].mode().fillna(0).iloc[0]  # Mode: most common value
            numeric_stats['iqr'] = numeric_stats['75%'] - numeric_stats['25%']  # Why: spread measurement
            numeric_stats['skew'] = df[numeric_cols].skew()       # Why: check symmetry of distribution
            numeric_stats['kurt'] = df[numeric_cols].kurt()       # Why: check tail heaviness

            st.markdown("### 📈 Numeric Summary Statistics")
            # Highlight max values for quick insight
            st.dataframe(numeric_stats.style.format("{:.2f}").highlight_max(axis=0, color="lightgreen"))

            # Plot histogram for each numeric column
            for col in numeric_cols:
                st.markdown(f"#### 📊 {col} Distribution")
                chart = alt.Chart(df).mark_bar().encode(
                    alt.X(col, bin=alt.Bin(maxbins=30), title=f"Distribution of {col}"),
                    y='count()'
                ).properties(width='container', height=200).interactive()
                st.altair_chart(chart, use_container_width=True)

    # ---- NON-NUMERIC COLUMNS ----
    if not non_numeric_cols.empty:
        with tab2:
            st.markdown("### 🔍 Categorical Data Insights")
            for col in non_numeric_cols:
                st.markdown(f"**📅 {col} Frequency**")
                # Frequency + percentage distribution
                freq_table = df[col].value_counts().reset_index()
                freq_table.columns = ['Category', 'Count']
                freq_table['Percentage'] = (freq_table['Count'] / len(df) * 100).round(2)
                st.table(freq_table.style.format({"Percentage": "{:.2f}%"}))

def handle_query_response(
    response: dict,
    db_name: str,
    db_type: str,
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None
) -> None:
    """Handles responses from query generation, displays results, and visualizations."""

    try:
        # Extract fields returned by SQL generator
        query = response.get("query", "")
        error = response.get("error", "")
        decision_log = response.get("decision_log", "")
        visualization_recommendation = response.get("visualization_recommendation", None)

        # Handle errors from query generation stage
        if error:
            displayed_error = generate_detailed_error_message(error)
            st.error(f"Error reason: {displayed_error}")
            return

        # If no query could be generated
        if not query:
            st.warning("No query generated. Please refine your message.")
            return

        # ✅ Show generated SQL query
        st.success("Query generated successfully")
        colored_header("SQL Query and Summary", color_name="blue-70", description="")
        st.code(query, language="sql")

        # Show decision-making log if available (transparency into AI reasoning)
        if decision_log:
            with st.expander("Decision Log", expanded=False):
                st.markdown(build_markdown_decision_log(decision_log))

        # Run SQL query against DB
        sql_results = get_data(query, db_name, db_type, host, user, password)

        # Handle no results case gracefully
        if sql_results.empty:
            no_result_reason = "The query executed successfully but did not match any records in the database."
            # More refined messages based on execution feedback
            if "no valid SQL query generated" in decision_log.get("execution_feedback", []):
                no_result_reason = "The query was not generated due to insufficient or ambiguous input."
            elif "SQL query validation failed" in decision_log.get("execution_feedback", []):
                no_result_reason = "The query failed validation checks and was not executed."
            st.warning(f"The query returned no results because: {no_result_reason}")
            return

        # Avoid confusing duplicate columns
        if sql_results.columns.duplicated().any():
            st.error("The query returned a DataFrame with duplicate column names. Please modify your query to avoid this.")
            return

        # Try converting object columns to datetime if possible (why: better visualizations & stats)
        for col in sql_results.select_dtypes(include=["object"]):
            try:
                sql_results[col] = pd.to_datetime(sql_results[col])
            except (ValueError, TypeError):
                pass

        # ✅ Show query results with filtering
        colored_header("Query Results and Filter", color_name="blue-70", description="")
        filtered_results = dataframe_explorer(sql_results, case=False)
        st.dataframe(filtered_results, use_container_width=True, height=600)

        # ✅ Show summary statistics
        colored_header("Summary Statistics and Export Options", color_name="blue-70", description="")
        display_summary_statistics(filtered_results)

        # ✅ Visualization Options (sidebar)
        if len(filtered_results.columns) >= 2:
            with st.sidebar.expander("📊 Visualization Options", expanded=True):
                # Identify numeric vs categorical columns
                numerical_cols = filtered_results.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = filtered_results.select_dtypes(include=["object", "category"]).columns.tolist()

                # Auto-suggest best X and Y columns
                suggested_x, suggested_y = None, None
                if numerical_cols:
                    suggested_x = numerical_cols[0]
                    suggested_y = numerical_cols[1] if len(numerical_cols) > 1 else (categorical_cols[0] if categorical_cols else None)
                elif categorical_cols:
                    suggested_x = categorical_cols[0]
                    suggested_y = categorical_cols[1] if len(categorical_cols) > 1 else None

                # Fallback defaults
                suggested_x = suggested_x or (filtered_results.columns[0] if not filtered_results.columns.empty else "Column1")
                suggested_y = suggested_y or (filtered_results.columns[1] if len(filtered_results.columns) > 1 else "Column2")

                # Mark suggested columns with ⭐ for clarity
                x_options = [f"{col} ⭐" if col == suggested_x else col for col in filtered_results.columns]
                y_options = [f"{col} ⭐" if col == suggested_y else col for col in filtered_results.columns]

                # Let user select X & Y columns for chart
                x_col = st.selectbox("Select X-axis Column", options=x_options, index=x_options.index(f"{suggested_x} ⭐"))
                y_col = st.selectbox("Select Y-axis Column", options=y_options, index=y_options.index(f"{suggested_y} ⭐"))

                # Choose chart type
                chart_type_options = ["None", "Bar Chart", "Line Chart", "Scatter Plot", "Area Chart", "Histogram"]
                chart_type = st.selectbox("Select Chart Type", options=chart_type_options)

                # Build chart if requested
                if chart_type != "None" and x_col and y_col:
                    chart = create_chart(filtered_results, chart_type, x_col.replace(" ⭐", ""), y_col.replace(" ⭐", ""))
                    if chart:
                        st.altair_chart(chart, use_container_width=True)

        # ✅ Export results to desired format
        export_format = st.selectbox("Select Export Format", options=["CSV", "Excel", "JSON"])
        export_results(filtered_results, export_format)

        # ✅ Save query to session history for later recall
        if "query_history" not in st.session_state:
            st.session_state.query_history = []
        st.session_state.query_history.append(query)

    except Exception as e:
        # Handle any unexpected runtime errors
        detailed_error = generate_detailed_error_message(str(e))
        st.error(f"An unexpected error occurred: {detailed_error}")
        logger.exception(f"Unexpected error: {e}")


def validate_sql_query(query:str)->bool:
    """validates the sql query if it is safe to execute"""
    if not isinstance(query,str):
        return False
    disallowed_keywords = r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|EXEC)\b'

    if re.search(disallowed_keywords, query, re.IGNORECASE):
        return False

    if not query.strip().lower().startswith(('select', 'with')):
        return False

    if query.count('(') != query.count(')'):
        return False

    return True

def export_results(sql_results: pd.DataFrame, export_format: str) -> None:
    """Exports the results to the selected format (CSV, Excel, or JSON)."""
    if export_format == "CSV":
        st.download_button(
            label="📥 Download Results as CSV",
            data=sql_results.to_csv(index=False),
            file_name='query_results.csv',
            mime='text/csv'
        )
    elif export_format == "Excel":
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            sql_results.to_excel(writer, index=False, sheet_name='Sheet1')
        excel_buffer.seek(0)    
        st.download_button(
            label="📥 Download Results as Excel",
            data=excel_buffer,
            file_name='query_results.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    elif export_format == "JSON":
        st.download_button(
            label="📥 Download Results as JSON",
            data=sql_results.to_json(orient='records'),
            file_name='query_results.json',
            mime='application/json'
        )
    else:
        st.error("⚠️ Selected export format is not supported.")


def analyze_dataframe_for_visualization(df: pd.DataFrame) -> list:
    """Analyzes the DataFrame and suggests suitable visualization types."""
    suggestions = set()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    logger.debug(f"Numerical Columns: {numerical_cols}")
    logger.debug(f"Categorical Columns: {categorical_cols}")

    if len(numerical_cols) == 1:
        suggestions.update(["Histogram", "Box Plot"])
    if len(categorical_cols) == 1:
        suggestions.update(["Bar Chart", "Pie Chart"])

    if len(numerical_cols) >= 2:
        suggestions.update(["Scatter Plot", "Line Chart"])
    elif len(numerical_cols) == 1 and len(categorical_cols) == 1:
        suggestions.update(["Bar Chart"])

    if len(numerical_cols) > 2:
        suggestions.add("Scatter Plot")

    time_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if time_cols:
        suggestions.add("Line Chart")

    ordered_suggestions = [chart for chart in SUPPORTED_CHART_TYPES.keys() if chart in suggestions]
    logger.debug(f"Ordered Suggestions: {ordered_suggestions}")
    return ordered_suggestions

def generate_detailed_error_message(error_message: str) -> str:
    """Generates a detailed and user-friendly explanation for the given error message."""
    try:
        prompt = f"Provide a detailed and user-friendly explanation for the following error message:\n\n{error_message}"
        detailed_error = get_completion_from_messages(SYSTEM_MESSAGE, prompt)
        return detailed_error.strip() if detailed_error else error_message
    except Exception as gen_err:
        logger.exception(f"Error generating detailed error message: {gen_err}")
        return error_message  # Fallback to the original error message

import speech_recognition as sr

def listen_for_query():
    """Function to listen to the user's voice and convert it to text with a 3-second timeout."""
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening for your query...")

        # Reduce the ambient noise adjustment time to 0.1 seconds
        recognizer.adjust_for_ambient_noise(source, duration=0.1)

        try:
            # Listen for a maximum of 3 seconds of speech
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)  # Timeout after 3 seconds of no speech
            
            # Convert the audio to text using Google Speech Recognition
            query = recognizer.recognize_google(audio)
            print(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return None
        except Exception as ex:
            print(f"An error occurred: {ex}")
            return None



# Database setup
db_type = st.sidebar.selectbox("Select Database Type 🔧", options=["SQLite", "PostgreSQL"])

# --- For SQLite Database ---
if db_type == "SQLite":
    # Upload SQLite database file (db, sqlite, sql extensions)
    uploaded_file = st.sidebar.file_uploader("📂 Upload SQLite Database", type=["db", "sqlite", "sql"])

    if uploaded_file:
        # Save uploaded file temporarily for processing
        db_file = save_temp_file(uploaded_file)
        # Extract schema info (tables + columns) from SQLite DB
        schemas = DB_Config.get_all_schemas(db_file, db_type='sqlite')
        table_names = list(schemas.keys())

        if not schemas:
            # Show error if no schema was found (maybe invalid file)
            st.error("🚨 Could not load any schemas, please check the database file.")

        if table_names:
            # Sidebar option to select tables (or select all)
            options = ["Select All"] + table_names
            selected_tables = st.sidebar.multiselect("📋 Select Tables", options=options, key="sqlite_tables")
            
            # Handle "Select All" option
            if "Select All" in selected_tables:
                selected_tables = table_names

            # Remove "Select All" placeholder from list if present
            selected_tables = [table for table in selected_tables if table != "Select All"]
            
            # Show selected tables in a colored header for clarity
            colored_header(f"🔍 Selected Tables: {', '.join(selected_tables)}", color_name="blue-70", description="")
            
            # Show schema details for each selected table in collapsible expanders
            for table in selected_tables:
                with st.expander(f"📖 View Schema: {table}", expanded=False):
                    st.json(schemas[table])

            # Input box for user to type SQL query (hidden label for clean UI)
            user_message = st.text_input(placeholder="💬 Type your SQL query here...", key="user_message", label="Your Query", label_visibility="hidden")
            
            # If user clicks "Speak" button, capture voice query
            if st.button("Speak:🎤"):
                user_message = listen_for_query()
         
            if user_message:
                # Collect only selected tables' schema to guide query generation
                selected_schemas = {table: schemas[table] for table in selected_tables}
                logger.debug(f"Schemas being passed to `generate_sql_query`: {selected_schemas}")
                
                # Generate SQL query using AI/NLP
                with st.spinner('🏎️ Generating SQL query...'):
                    response = generate_sql_query(user_message, selected_schemas, db_name=db_file, db_type='sqlite')
                
                # Execute and display results of query
                handle_query_response(response, db_file, db_type='sqlite')

        else:
            st.info("📭 No tables found in the database.")
    else:
        st.info("📥 Please upload a database file to start.")

# --- For PostgreSQL Database ---
elif db_type == "PostgreSQL":
    # Sidebar inputs for PostgreSQL connection details
    with st.sidebar.expander("🔐 PostgreSQL Connection Details", expanded=True):
        postgres_host = st.text_input("🏠 Host", placeholder="PostgreSQL Host")
        postgres_db = st.text_input("🗄️ DB Name", placeholder="Database Name")
        postgres_user = st.text_input("👤 Username", placeholder="Username")
        postgres_password = st.text_input("🔑 Password", type="password", placeholder="Password")

    # Ensure all connection details are filled in
    if all([postgres_host, postgres_db, postgres_user, postgres_password]):
        # Fetch schemas (tables + columns) from PostgreSQL database
        schemas = DB_Config.get_all_schemas(postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)
        table_names = list(schemas.keys())

        if table_names:
            # Sidebar option to select tables (or select all)
            options = ["Select All"] + table_names
            selected_tables = st.sidebar.multiselect("📋 Select Tables", options=options, key="postgresql_tables")
            
            if "Select All" in selected_tables:
                selected_tables = table_names

            selected_tables = [table for table in selected_tables if table != "Select All"]
            
            # Show selected tables
            colored_header(f"🔍 Selected Tables: {', '.join(selected_tables)}", color_name="blue-70", description="")
            
            # Show schema details for each selected table
            for table in selected_tables:
                with st.expander(f"📖 View Schema: {table}", expanded=False):
                    st.json(schemas[table])

            # Input for natural language SQL query
            user_message = st.text_input(placeholder="💬 Type your SQL query here...", key="user_message_pg", label="Your Query", label_visibility="hidden")
            
            if user_message:
                with st.spinner('🏎️ Generating SQL query...'):
                    # Collect selected table schemas
                    selected_schemas = {table: schemas[table] for table in selected_tables}
                    logger.debug(f"Schemas being passed to `generate_sql_query`: {selected_schemas}")
                    
                    # Generate SQL for PostgreSQL
                    response = generate_sql_query(user_message, selected_schemas, db_name=postgres_db, db_type='postgresql')
                
                # Execute query and display results
                handle_query_response(response, postgres_db, db_type='postgresql', host=postgres_host, user=postgres_user, password=postgres_password)
        else:
            st.info("📭 No tables found in the database.")
    else:
        # Warn user if connection details are missing
        st.info("🔒 Please fill in all PostgreSQL connection details to start.")
