import os
import google.auth
from google.cloud import aiplatform
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

# Fetch the API key and model from the environment
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

# Initialize Google Cloud AI Platform client
def initialize_client():
    credentials, project = google.auth.load_credentials_from_file(GEMINI_API_KEY)  # You can directly pass the API key
    aiplatform.init(credentials=credentials)

def generate_sql(natural_language_query):
    # Initialize the client if it is not initialized yet
    initialize_client()

    # Define the model and endpoint (adjust if needed)
    model_endpoint = f"projects/my-nlp-project/locations/us-central1/models/gemini-1.5-flash"

    
    # Create a prediction client
    model = aiplatform.gapic.PredictionServiceClient()

    # Generate the response from Gemini 1.5
    response = model.predict(
        endpoint=model_endpoint,
        instances=[{"content": natural_language_query}]
    )

    # Extract the generated SQL query
    sql_query = response.predictions[0]  # Assuming the response is in this structure

    return sql_query
