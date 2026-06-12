from typing import List
from .utils import handle_error, json_file_read
from .schema import Prompt, FunctionDef
from pydantic import ValidationError


def load_prompts(file_path: str) -> List[Prompt]:
    """Load prompts from a JSON file.

    Args:
        file_path: Path to the prompts JSON file.

    Returns:
        List of Prompt objects.
    """
    data = json_file_read(file_path)
    try:
        return [Prompt(**item) for item in data]
    except ValidationError as error:
        handle_error(f"Error: Invalid data in {file_path}: {error}")


def load_functions(file_path: str) -> List[FunctionDef]:
    """Load function definitions from a JSON file.

    Args:
        file_path: Path to the function definitions JSON file.

    Returns:
        List of FunctionDef objects.
    """
    data = json_file_read(file_path)
    try:
        functions = [FunctionDef(**item) for item in data]
    except ValidationError as error:
        handle_error(f"Error: Invalid data in {file_path}: {error}")
    if not functions:
        handle_error(f"Error: No functions defined in {file_path}")
    return functions
