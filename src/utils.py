import sys
import json
import os
from typing import NoReturn, List, Any, Dict, Tuple
from .schema import FunctionCall
from llm_sdk import Small_LLM_Model  # type: ignore


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


def load_vocab(
        model: Small_LLM_Model) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Build vocabulary dictionaries from the model's vocab file.

    Args:
        model: The LLM model instance.

    Returns:
        A tuple of (token_to_id, id_to_token) dictionaries.
    """
    path = model.get_path_to_vocab_file()
    token_to_id = json_file_read(path)
    id_to_token = {value: key for key, value in token_to_id.items()}
    return token_to_id, id_to_token


def build_number_token_ids(id_to_token: Dict[int, str]) -> List[int]:
    """Precompute vocabulary tokens valid during number generation.

    Args:
        id_to_token: Mapping from token ID to raw vocabulary token string.

    Returns:
        List of token IDs whose text consists only of digits, '.', or
        '-'.
    """
    safe_chars = "0123456789" + ".-" + "]},"
    return ([token_id for token_id, text in id_to_token.items()
             if text and all(ch in safe_chars for ch in text)])


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
