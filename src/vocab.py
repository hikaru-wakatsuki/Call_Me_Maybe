from llm_sdk import Small_LLM_Model  # type: ignore
from typing import Dict, Tuple, List
from .utils import json_file_read, handle_error


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


def decode_custom(ids: List[int], id_to_token: Dict[int, str]) -> str:
    """Decode token IDs back into text without using the SDK's decode.

    Args:
        ids: List of token IDs to decode.
        id_to_token: Mapping from token ID to raw vocabulary token string.

    Returns:
        The decoded text.
    """
    try:
        raw = "".join(id_to_token[id] for id in ids)
    except KeyError as e:
        handle_error(f"Error: Unknown token ID encountered during decode: {e}")
    return raw.replace("Ġ", " ")


def _split_into_chunks(marked: str) -> List[str]:
    """Split text on the Ġ space marker into word-level chunks.

    Args:
        marked: Text with spaces already replaced by Ġ.

    Returns:
        List of chunks, each starting with Ġ except possibly the first.
    """
    parts = marked.split("Ġ")
    chunks = [parts[0]] if parts[0] else []
    chunks.extend("Ġ" + p for p in parts[1:])
    return chunks


def _longest_match(text: str, token_to_id: Dict[str, int]) -> str:
    """Find the longest vocab token that is a prefix of text.

    Args:
        text: The remaining text to match against (a single chunk).
        token_to_id: Mapping from vocab token string to ID.

    Returns:
        The longest matching token string.
    """
    for length in range(len(text), 0, -1):
        candidate = text[:length]
        if candidate in token_to_id:
            return candidate
    handle_error(f"Error: No matching token found for text: {text!r}")


def _encode_chunk(remaining: str, token_to_id: Dict[str, int]) -> List[int]:
    """Encode a single word-level chunk into token IDs.

    Args:
        remaining: A word-level chunk (e.g. "Ġis").
        token_to_id: Mapping from vocab token string to ID.

    Returns:
        List of token IDs covering the entire chunk.
    """
    ids: List[int] = []
    while remaining:
        match = _longest_match(remaining, token_to_id)
        ids.append(token_to_id[match])
        remaining = remaining[len(match):]
    return ids


def encode_custom(text: str, token_to_id: Dict[str, int]) -> List[int]:
    """Encode text into token IDs without using the SDK's encode.

    Args:
        text: The text to encode.
        token_to_id: Mapping from vocab token string to ID.

    Returns:
        List of token IDs for the text.
    """
    marked = text.replace(" ", "Ġ")
    chunks = _split_into_chunks(marked)
    ids: List[int] = []
    for chunk in chunks:
        ids.extend(_encode_chunk(chunk, token_to_id))
    return ids
