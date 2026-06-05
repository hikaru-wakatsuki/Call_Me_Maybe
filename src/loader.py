from typing import List
from .utils import handle_error, json_file_read
from .schema import Prompt, FunctionDef
from pydantic import ValidationError


def load_prompts(file_path: str) -> List[Prompt]:
    data = json_file_read(file_path)
    try:
        return [Prompt(**item) for item in data]
    except ValidationError as error:
        handle_error(f"Error: Invalid data in {file_path}: {error}")


def load_functions(file_path: str) -> List[FunctionDef]:
    data = json_file_read(file_path)
    try:
        return [FunctionDef(**item) for item in data]
    except ValidationError as error:
        handle_error(f"Error: Invalid data in {file_path}: {error}")
