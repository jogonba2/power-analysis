import logging
import sys

COLORS = {
    "grey": "\x1b[38;20m",
    "yellow": "\x1b[33;20m",
    "bold_yellow": "\x1b[33;1m",
    "red": "\x1b[31;20m",
    "bold_red": "\x1b[31;1m",
    "green": "\x1b[38;5;10m",
    "blue": "\x1b[38;5;27m",
    "reset": "\x1b[0m",
}


def log(logger_method, text: str, color: str) -> str:
    """
    Add color to a log text.

    Args:
        logger_method: a logger method, e.g., "info" or "warn"
        text (str): a text.
        color (str): a color in `COLORS`

    Returns:
        str: a text with color codes added.
    """
    return logger_method(COLORS[color] + text + COLORS["reset"])


def get_logger(module_name: str) -> logging.Logger:
    """
    Returns the logger used across TextMachina modules.

    Args:
        module_name (str): name of the module.

    Returns:
        logging.Logger: the logger.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    return logger
