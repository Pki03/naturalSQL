import os
from dotenv import load_dotenv
import google.generativeai as genai
import re

# Load environment variables from a .env file
load_dotenv()

import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")


# Raise an error if the API key is not set, as it's required for API interaction
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Configure the Generative AI client with the provided API key
genai.configure(api_key=GEMINI_API_KEY)

def get_completion_from_messages(
    system_message: str,
    user_message: str,
    temperature: float = 0.3,
) -> str:
    """
    Generates a completion response from the Generative AI model based on system and user messages.

    Args:
        system_message (str): The system-provided message for context or instruction.
        user_message (str): The user's input query.
        temperature (float): Controls the randomness of the output. Lower values make it more deterministic.

    Returns:
        str: The model's response or an error message in case of failure.
    """
    try:
        # Combine system and user messages into a single string to provide context to the model
        combined_message = f"{system_message}\n\nUser Query: {user_message}"

        # Instantiate the Generative AI model using the specified Gemini model
        model = genai.GenerativeModel(GEMINI_MODEL)

        # Generate content using the combined message with specified temperature
        response = model.generate_content(
            contents=combined_message,
            generation_config={"temperature": temperature}
        )

        # Extract and clean the generated text from the response
        text = response.text if isinstance(response.text, str) else str(response.text)
        clean_text = re.sub(r'```json\n|\n```', '', text)  # Remove unwanted formatting if present

        return clean_text
    except Exception as e:
        # Handle errors gracefully and return an error message
        error_msg = f"Error generating response: {str(e)}"
        return error_msg
