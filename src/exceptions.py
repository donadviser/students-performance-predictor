import sys
from src.logger import logger

def get_error_details1(error: Exception, error_details:sys):
    """
    Returns the error message and error details.

    Args:
        error (str): The error message.
        error_details (sys): The error details.

    Returns:
        str: A formated string containing the error filename, line number, and message.
    """

    try:
        _, _, exc_tb = error_details.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = str(error)
        return f"Error occured in Python Script: {file_name}, Line: {line_number}, Message: {error_message}"
    except (AttributeError, NameError):
        return f"Error: Unable to retrieve detailed error information: {str(error)}"
    

class CustomException1(Exception):
    """
    Custom exception class.

    Args: 
        error_message (str): The error message
        error_details (sys): The error details

    Returns:
        str: A formated string containing the error filename, line number, and message.
        
    Usage:
        CustomException(e, sys)
    """

    def __init__(self, error_message: str, error_details:sys):
        super().__init__(error_message)
        self.error_message = get_error_details1(error_message, error_details=error_details)
        logger.info(self.error_message)
    
    def __str__(self):
        return self.error_message
    


def get_error_details(error_details: Exception, error: str=None) -> str:
    """
    Returns the error message and error details.

    Usage: CustomException(e, "HARD ERROR")

    Args:
        error (str): The error message.
        error_details (Exception): The exception object containing details.

    Returns:
        str: A formatted string containing the error filename, line number, and message.
    """

    try:
        # Access traceback information from the exception object
        exc_info = error_details.__traceback__
        file_name = exc_info.tb_frame.f_code.co_filename
        line_number = exc_info.tb_lineno
        if not error:
            error_message = str(error_details)
        else:
            error_message = f"{str(error_details)} | USER INPUT: {str(error)}"
        return f"Error occured in Python Script: {file_name}, Line: {line_number}, Message: {error_message}"
    except (AttributeError, NameError):
        # Handle potential missing attributes
        return f"Error: Unable to retrieve detailed error information: {str(error)}"

class CustomException(Exception):
    """
    Custom exception class.

    Args: 
        error_message (str): The error message
        error_details (sys): The error details

    Returns:
        str: A formated string containing the error filename, line number, and message.
        
    Usage:
        CustomException(e, "user error message")
    """

    def __init__(self, error_details: Exception, error_message: str=None):
        super().__init__(error_message)
        self.error_message = get_error_details(error_details, error_message)
        logger.info(self.error_message)

    def __str__(self):
        return self.error_message