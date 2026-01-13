# Main entry point for the RAG Chatbot application.
# Run this file to start the Gradio interface.

import logging
from frontend import launch_app
import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, 'INFO'),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting RAG Chatbot application...")
    try:
        launch_app()
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed to start: {e}")
        raise