from typing import Optional
import functools
import pandas as pd
from termcolor import colored
import globals.constants as constants

def _compile_subject_args(trial: Optional[pd.Series] = None, subject: Optional[str] = None, trial_number: Optional[int] = None, mvc_level: Optional[int] = None) -> tuple:
    """Compiles subject specific arguments into a tuple for use in logging messages."""
    if trial is None and (subject is None or trial_number is None or mvc_level is None):
        raise ValueError("Either a trial series or subject specific information must be provided.")
    
    if trial is None:
        return (subject, trial_number, mvc_level)
    
    if not all([key in trial.index for key in ["subject", "trial_number", "mvc_level"]]):
        raise ValueError("Trial series must contain 'subject', 'trial_number', and 'mvc_level' keys.")

    return (trial["subject"], trial["trial_number"], trial["mvc_level"])

class Log:
    """Project wide logging utility."""

    @staticmethod
    def _logger(func):
        """Decorator for logging messages. Configured through constants and accessed through Log class methods."""
        @functools.wraps(func)
        def wrapper_logger(*args, **kwargs):
            identifier = func.__name__

            if identifier not in constants.logging:
                return
            
            message = args[0] # message is always first argument

            # compile subject specific information if present
            if len(kwargs) != 0:
                contains_tab = '\t' in message
                message = f"{'\t' if contains_tab else ''}{format_subject(**kwargs)} {message.replace('\t', '')}"
            
            color = constants.logging_colors[identifier]
            print(colored(message, color))

            value = func(*args, **kwargs)
            return value
        return wrapper_logger

    @_logger
    @staticmethod
    def info(m: str, trial: Optional[pd.Series] = None, subject: Optional[str] = None, trial_number: Optional[int] = None, mvc_level: Optional[int] = None):
        pass
    
    @_logger
    @staticmethod
    def error(m: str, trial: Optional[pd.Series] = None, subject: Optional[str] = None, trial_number: Optional[int] = None, mvc_level: Optional[int] = None):
        pass

    @_logger
    @staticmethod
    def debug(m: str, trial: Optional[pd.Series] = None, subject: Optional[str] = None, trial_number: Optional[int] = None, mvc_level: Optional[int] = None):
        pass

    @_logger
    @staticmethod
    def success(m: str, trial: Optional[pd.Series] = None, subject: Optional[str] = None, trial_number: Optional[int] = None, mvc_level: Optional[int] = None):
        pass

    @_logger
    @staticmethod
    def warn(m: str, trial: Optional[pd.Series] = None, subject: Optional[str] = None, trial_number: Optional[int] = None, mvc_level: Optional[int] = None):
        pass

def format_subject(
        trial: Optional[pd.Series] = None, 
        subject: Optional[str] = None, 
        trial_number: Optional[int] = None, 
        mvc_level: Optional[int] = None, 
        leading: bool = True, 
        verbose: bool = False
    ) -> str:
    """Formats a subject string for use in various contexts. Either a trial series or subject specific information must be provided."""
    subject, trial_number, mvc_level = _compile_subject_args(trial, subject, trial_number, mvc_level)
    formatted = f"{subject}.{mvc_level}.{trial_number}"

    if verbose:
        formatted = f"{subject} | Trial {trial_number} @ {mvc_level}% MVC"

    if leading:
        return f"[{formatted}]"

    return formatted