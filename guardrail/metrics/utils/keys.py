import os
import openai
from dotenv import load_dotenv, find_dotenv

def init_openai_key():
    # Load environment variables from .env file
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    # Retrieve your OpenAI API key from the environment variables
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai.api_key = openai_api_key

    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables.")

    return openai_api_key

def init_guardrail_key():
    # Load environment variables from .env file
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    # Retrieve your OpenAI API key from the environment variables
    guardrail_api_key = os.getenv("GUARDRAIL_API_KEY")
    openai.api_key = guardrail_api_key

    if not guardrail_api_key:
        raise ValueError("OpenAI API key not found in environment variables.")

    return guardrail_api_key