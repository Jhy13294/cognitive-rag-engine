import logging

from api_client import APIClient, APIError
from config import Config
from logger import setup_logger

logger = setup_logger(__name__, level=logging.INFO)


def main():
    """Run the command-line chat flow."""
    logger.info("=" * 50)
    logger.info("Application started")

    try:
        Config.validate()
        logger.info("Configuration validation passed")

        client = APIClient(Config.API_KEY, Config.API_URL)
        logger.info("API client initialized")

        message = input("Enter your question: ")
        logger.info("User input: %s...", message[:50])

        response = client.chat(message)
        content = response["choices"][0]["message"]["content"]

        print("\n" + "=" * 50)
        print(content)
        print("=" * 50)

        logger.info("Response completed | content_length=%s", len(content))

    except APIError as e:
        logger.error("API error: %s", e)
        print(e)

    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        print(f"Unexpected error: {e}")

    finally:
        logger.info("Application finished")
        logger.info("=" * 50)


if __name__ == "__main__":
    main()
