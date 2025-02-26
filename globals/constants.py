from termcolor._types import Color

logging = ['info', 'success', 'warn', 'error', 'debug'] # select forms of logging (info, warn, error, success, debug)
logging_colors: dict[str, Color] = {
    'info': 'white',
    'success': 'green',
    'warn': 'yellow',
    'error': 'red',
    'debug': 'blue'
} # colors for logging

sampling_frequency = 2048 # sampling frequency of the data in Hz