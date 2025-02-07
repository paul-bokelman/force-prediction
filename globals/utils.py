from termcolor import colored
import globals.constants as constants

class Log:
    """Project wide logging utility."""
    @staticmethod
    def info(message):
        if 'info' not in constants.logging:
            return

        print(message)

    @staticmethod
    def error(message):
        if 'error' not in constants.logging:
            return

        print(colored(message, 'red'))

    @staticmethod
    def success(message):
        if 'success' not in constants.logging:
            return

        print(colored(message, 'green'))
    
    @staticmethod
    def warn(message):
        if 'warn' not in constants.logging:
            return
        
        print(colored(message, 'yellow'))