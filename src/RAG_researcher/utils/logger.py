
import logging
from datetime import datetime


class customLogger():
    ''' Creates a logger for the RAG application to track user queries and system responses.'''
    current_datetime = datetime.now()
    formatted_date = current_datetime.strftime("%Y-%m-%d")


    # Initialize the logger
    custom_logger = logging.getLogger('RAG_logger')
    custom_logger.setLevel(logging.INFO)

    # Prevent this logger from being passed to the root logger and conaminating file
    custom_logger.propagate = False

    # Create a file handler for your custom logs
    file_handler = logging.FileHandler(f'RAG_{formatted_date}.log')
    file_handler.setLevel(logging.INFO)

    # Format the output of the logs
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    file_handler.setFormatter(formatter)

    custom_logger.addHandler(file_handler)

rag_logger = customLogger.custom_logger