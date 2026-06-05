from typing import NoReturn, List, Any
import sys
import json


def handle_error(message: str) -> NoReturn:
    print(message)
    sys.exit(1)


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
