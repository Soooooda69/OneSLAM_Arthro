import logging

def configure_logging():
    # Create a logger
    logger = logging.getLogger()
    
    if not logger.handlers:
        # Set the logging level
        logger.setLevel(logging.INFO)

        # Create a file handler and set the level to DEBUG
        file_handler = logging.FileHandler('logfile.txt')
        file_handler.setLevel(logging.INFO)

        # Create a formatter and add it to the file handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)
# Call the function to configure logging when this module is imported
configure_logging()
