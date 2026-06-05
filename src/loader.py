from typing import List, Any
import json
from .utils import handle_error
from .schema import Prompt, FunctionDef
from pydantic import ValidationError


def json_file_read(file_path: str) -> List[Any]:
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        handle_error(f"Error: File not found: {file_path}")
    except PermissionError:
        handle_error(f"Error: Permission denied: {file_path}")
    except json.JSONDecodeError as error:
        handle_error(f"Error: Invalid JSON in {file_path}: {error}")


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
