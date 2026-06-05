from typing import NoReturn
import sys


def handle_error(message: str) -> NoReturn:
    print(message)
    sys.exit(1)
