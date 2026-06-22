import json
from llm_sdk import Small_LLM_Model  # type: ignore
from typing import List, Dict, Callable
from .schema import FunctionDef, Prompt, TypeDef, FunctionCall
from .encode import encode_custom
from .decode import decode_custom, append_tokens


MAX_STEPS = 100


def _build_parameters_prompt(request: Prompt, function: FunctionDef) -> str:
    """Build a prompt for argument generation.

    Args:
        request: The user's prompt.
        function: The selected function definition.

    Returns:
        A prompt string for argument generation.
    """
    prompt = f"User request: {request.prompt}"
    params = ", ".join(
        f"{k}: {v.type}" for k, v in function.parameters.items())
    prompt += f"\nFunction: {function.name}({params}): {function.description}"
    prompt += "\nGenerate the arguments as JSON for the function call."
    prompt += "\nArguments:"
    return prompt


def _generate_primitive_ids(
        model: Small_LLM_Model, input_ids: List[int],
        id_to_token: Dict[int, str],
        visualize: bool = False) -> List[int]:
    """Generate token IDs for a primitive value (number, boolean, etc.).

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        id_to_token: Mapping from token ID to raw vocabulary token string.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        List of generated token IDs for the primitive value.
    """
    generated: List[int] = []
    for _ in range(MAX_STEPS):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = logits.index(max(logits))
        token = decode_custom([next_id], id_to_token)
        if ',' in token or '}' in token or ']' in token:
            break
        append_tokens(generated, [next_id], id_to_token, visualize)
    return generated


def _generate_string_ids(
        model: Small_LLM_Model, input_ids: List[int],
        id_to_token: Dict[int, str],
        encode_cached: Callable[[str], List[int]],
        visualize: bool = False) -> List[int]:
    """Generate token IDs for a string value, wrapped in double quotes.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        id_to_token: Mapping from token ID to raw vocabulary token string.
        encode_cached: Cached encoder for fixed/repeating text.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        List of generated token IDs for the string value.
    """
    generated: List[int] = []
    append_tokens(generated, encode_cached('"'), id_to_token, visualize)
    for _ in range(MAX_STEPS):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = logits.index(max(logits))
        token = decode_custom([next_id], id_to_token)
        if '"' in token:
            break
        append_tokens(generated, [next_id], id_to_token, visualize)
    append_tokens(generated, encode_cached('"'), id_to_token, visualize)
    return generated


def _generate_array_ids(
        model: Small_LLM_Model, input_ids: List[int],
        typedef: TypeDef, id_to_token: Dict[int, str],
        encode_cached: Callable[[str], List[int]],
        visualize: bool = False) -> List[int]:
    """Generate token IDs for an array value.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        typedef: The type definition of the array elements.
        id_to_token: Mapping from token ID to raw vocabulary token string.
        encode_cached: Cached encoder for fixed/repeating text.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        List of generated token IDs for the array value.
    """
    generated: List[int] = []
    append_tokens(generated, encode_cached('['), id_to_token, visualize)
    for _ in range(MAX_STEPS):
        value_ids = _generate_value_ids(
            model, input_ids + generated, typedef,
            id_to_token, encode_cached, visualize)
        generated.extend(value_ids)
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = logits.index(max(logits))
        token = decode_custom([next_id], id_to_token)
        if ']' in token:
            break
        append_tokens(generated, encode_cached(','), id_to_token, visualize)
    append_tokens(generated, encode_cached(']'), id_to_token, visualize)
    return generated


def _generate_value_ids(
        model: Small_LLM_Model, input_ids: List[int],
        typedef: TypeDef, id_to_token: Dict[int, str],
        encode_cached: Callable[[str], List[int]],
        visualize: bool = False) -> List[int]:
    """Generate value tokens for a function argument using constrained
       decoding.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        typedef: The type definition of the argument.
        id_to_token: Mapping from token ID to raw vocabulary token string.
        encode_cached: Cached encoder for fixed/repeating text.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        List of generated token IDs for the value.
    """
    if typedef.properties is not None:
        return _generate_parameters_ids(
            model, input_ids, typedef.properties,
            id_to_token, encode_cached, visualize)
    elif typedef.items is not None:
        return _generate_array_ids(
            model, input_ids, typedef.items,
            id_to_token, encode_cached, visualize)
    elif typedef.type in ("string", "str"):
        return _generate_string_ids(
            model, input_ids, id_to_token, encode_cached, visualize)
    else:
        return _generate_primitive_ids(
            model, input_ids, id_to_token, visualize)


def _generate_parameters_ids(model: Small_LLM_Model, input_ids: List[int],
                             parameters: Dict[str, TypeDef],
                             id_to_token: Dict[int, str],
                             encode_cached: Callable[[str], List[int]],
                             visualize: bool = False) -> List[int]:
    """Generate arguments for a function call using constrained decoding.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        parameters: Dictionary mapping parameter names to type definitions.
        id_to_token: Mapping from token ID to raw vocabulary token string.
        encode_cached: Cached encoder for fixed/repeating text.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        List of generated token IDs for the parameters.
    """
    generated: List[int] = []
    append_tokens(generated, encode_cached('{'), id_to_token, visualize)
    params = list(parameters.items())
    for i, (key, typedef) in enumerate(params):
        append_tokens(
            generated, encode_cached(f'"{key}": '), id_to_token, visualize)
        value_ids = _generate_value_ids(
            model, input_ids + generated, typedef,
            id_to_token, encode_cached, visualize)
        generated.extend(value_ids)
        if i < len(params) - 1:
            append_tokens(
                generated, encode_cached(', '), id_to_token, visualize)
    append_tokens(generated, encode_cached('}'), id_to_token, visualize)
    return generated


def generate_function_call(model: Small_LLM_Model, request: Prompt,
                           function: FunctionDef,
                           token_to_id: Dict[str, int],
                           id_to_token: Dict[int, str],
                           encode_cached: Callable[[str], List[int]],
                           visualize: bool = False) -> FunctionCall:
    """Generate a function call from a natural language prompt.

    Args:
        model: The LLM model instance.
        request: The user's prompt.
        function: The selected function definition.
        token_to_id: Mapping from vocab token string to ID.
        id_to_token: Mapping from token ID to raw vocabulary token string.
        encode_cached: Cached encoder for fixed/repeating text.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        A FunctionCall object containing the function name and arguments.
    """
    if visualize:
        print("Generating arguments: ", end='', flush=True)
    parameters_prompt = _build_parameters_prompt(request, function)
    input_ids = encode_custom(parameters_prompt, token_to_id)
    parameters_ids = _generate_parameters_ids(
        model, input_ids, function.parameters,
        id_to_token, encode_cached, visualize)
    if visualize:
        print()
    parameters = json.loads(decode_custom(parameters_ids, id_to_token))
    return FunctionCall(
        prompt=request.prompt, name=function.name, parameters=parameters)
