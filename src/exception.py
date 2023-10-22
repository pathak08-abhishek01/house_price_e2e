import sys
from logger import logging

def error_message_details(error, error_detail: sys):
    """
    Generate the error message details based on the error and error_detail.

    Parameters:
        error (Exception): The error that occurred.
        error_detail (sys): The error detail.

    Returns:
        str: The error message containing the name of the python script, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """
        Initializes a new instance of the class.

        Parameters:
            error_message (str): The error message.
            error_detail (sys): The error detail object.

        Returns:
            None
        """
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        """
        Returns a string representation of the object.

        :return: A string representing the error message.
        :rtype: str
        """
        return self.error_message
    
