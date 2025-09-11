import os
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env file
load_dotenv()

# Fetch the OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)


def get_response(input_text, model="gpt-5-nano"):
    """
        Function to get a response from the OpenAI API.

        Args:
        - input_text (str): The input text/question to be sent to the model.
        - model (str): The OpenAI model to use, default is 'gpt-5'.

        Returns:
        - str: The output text from the model.
    """
    try:
        response = client.responses.create(
            model=model,
            input=input_text
        )
        return response.output_text  # Extracting the output text from the response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
