import ipywidgets as widgets
from IPython.display import display
import logging

DEBUG_OUTPUT_WIDGET = widgets.Output(layout={
    'flex': '0 1 640px',
    'width': '100%',
    'border': '1px solid black',
    'overflow': 'scroll'
})

class OutputWidgetHandler(logging.Handler):
    """ Custom logging handler sending logs to an output widget """

    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        self.out = DEBUG_OUTPUT_WIDGET

    def emit(self, record):
        """ Overload of logging.Handler method """
        formatted_record = self.format(record)
        new_output = {
            'name': 'stdout',
            'output_type': 'stream',
            'text': formatted_record+'\n'
        }
        self.out.outputs = (new_output, ) + self.out.outputs

    def show_logs(self):
        """ Show the logs """
        display(self.out)

    def clear_logs(self):
        """ Clear the current logs """
        self.out.clear_output()

        
DEFAULT_OUTPUT_HANDLER = OutputWidgetHandler()
DEFAULT_OUTPUT_HANDLER.setFormatter(logging.Formatter('%(asctime)s  - [%(levelname)s] %(message)s'))

def getLogger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addHandler(DEFAULT_OUTPUT_HANDLER)
    logger.setLevel(logging.INFO)
    return logger


def add_output_handler(logger: logging.Logger) -> logging.Logger:
    logger.addHandler(DEFAULT_OUTPUT_HANDLER)
    logger.setLevel(logging.INFO)
    return logger


def debug_capture(*args, **kwargs):
    return DEBUG_OUTPUT_WIDGET.capture(*args, **kwargs)


def show_logs():
    DEFAULT_OUTPUT_HANDLER.show_logs()