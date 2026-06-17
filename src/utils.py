import sys
import json
import os
from typing import NoReturn, List, Any
from functools import lru_cache
from llm_sdk import Small_LLM_Model
from .schema import FunctionCall


def handle_error(message: str) -> NoReturn:
    """Print an error message and exit the program.

    Args:
        message: The error message to display.
    """
    print(message)
    sys.exit(1)


def json_file_read(file_path: str) -> Any:
    """Read and parse a JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed JSON content .
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        handle_error(f"Error: File not found: {file_path}")
    except PermissionError:
        handle_error(f"Error: Permission denied: {file_path}")
    except json.JSONDecodeError as error:
        handle_error(f"Error: Invalid JSON in {file_path}: {error}")


def write_output(file_path: str, results: List[FunctionCall]) -> None:
    """Write function call results to a JSON file.

    Args:
        file_path: Path to the output JSON file.
        results: List of FunctionCall objects to write.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump([r.model_dump() for r in results], f, indent=4)
    except PermissionError:
        handle_error(f"Error: Permission denied: {file_path}")
    except OSError as e:
        handle_error(f"Error: Failed to write output: {e}")


@lru_cache(maxsize=None)
def encode_cached(model: Small_LLM_Model, text: str) -> List[int]:
    """Encode a fixed/repeating string, caching the result per model.

    Args:
        model: The LLM model instance whose tokenizer is used.
        text: The fixed text to encode.

    Returns:
        The list of token IDs for ``text``.
    """
    return model.encode(text).tolist()[0]
