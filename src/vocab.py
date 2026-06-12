from llm_sdk import Small_LLM_Model
from typing import Dict, Tuple
from .utils import json_file_read


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
