from typing import Dict, List
from .utils import handle_error


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


def append_tokens(generated: List[int], token_ids: List[int],
                  id_to_token: Dict[int, str],
                  visualize: bool = False) -> None:
    """Append token IDs to the generated sequence, optionally printing them.

    Args:
        generated: The list of generated token IDs, extended in place.
        token_ids: The token IDs to append.
        id_to_token: Mapping from token ID to raw vocabulary token string.
        visualize: Whether to print the decoded text to the terminal.
    """
    generated.extend(token_ids)
    if visualize:
        print(decode_custom(token_ids, id_to_token), end='', flush=True)
