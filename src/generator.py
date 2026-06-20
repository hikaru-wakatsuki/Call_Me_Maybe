import json
from llm_sdk import Small_LLM_Model  # type: ignore
from typing import List, Dict
from .schema import FunctionDef, Prompt, TypeDef, FunctionCall
from .utils import encode_cached


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
        visualize: bool = False) -> List[int]:
    """Generate token IDs for a primitive value (number, boolean, etc.).

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        List of generated token IDs for the primitive value.
    """
    generated: List[int] = []
    for _ in range(MAX_STEPS):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = logits.index(max(logits))
        token = model.decode([next_id])
        if ',' in token or '}' in token or ']' in token:
            break
        if visualize:
            print(token, end='', flush=True)
        generated.append(next_id)
    return generated


def _generate_string_ids(
        model: Small_LLM_Model, input_ids: List[int],
        visualize: bool = False) -> List[int]:
    """Generate token IDs for a string value, wrapped in double quotes.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        List of generated token IDs for the string value.
    """
    generated: List[int] = []
    generated.extend(encode_cached(model, '"'))
    for _ in range(MAX_STEPS):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = logits.index(max(logits))
        token = model.decode([next_id])
        if '"' in token:
            break
        if visualize:
            print(token, end='', flush=True)
        generated.append(next_id)
    generated.extend(encode_cached(model, '"'))
    return generated


def _generate_array_ids(
        model: Small_LLM_Model, input_ids: List[int],
        typedef: TypeDef, visualize: bool = False) -> List[int]:
    """Generate token IDs for an array value.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        typedef: The type definition of the array elements.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        List of generated token IDs for the array value.
    """
    generated: List[int] = []
    generated.extend(encode_cached(model, '['))
    for _ in range(MAX_STEPS):
        value_ids = _generate_value_ids(model, input_ids + generated, typedef)
        generated.extend(value_ids)
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = logits.index(max(logits))
        token = model.decode([next_id])
        if ']' in token:
            break
        if visualize:
            print(token, end='', flush=True)
        generated.extend(encode_cached(model, ','))
    generated.extend(encode_cached(model, ']'))
    return generated


def _generate_value_ids(
        model: Small_LLM_Model, input_ids: List[int],
        typedef: TypeDef, visualize: bool = False) -> List[int]:
    """Generate value tokens for a function argument using constrained
       decoding.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        typedef: The type definition of the argument.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        List of generated token IDs for the value.
    """
    if typedef.properties is not None:
        return _generate_parameters_ids(
            model, input_ids, typedef.properties, visualize)
    elif typedef.items is not None:
        return _generate_array_ids(model, input_ids, typedef.items, visualize)
    elif typedef.type in ("string", "str"):
        return _generate_string_ids(model, input_ids, visualize)
    else:
        return _generate_primitive_ids(model, input_ids, visualize)


def _generate_parameters_ids(model: Small_LLM_Model, input_ids: List[int],
                             parameters:  Dict[str, TypeDef],
                             visualize: bool = False) -> List[int]:
    """Generate arguments for a function call using constrained decoding.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        parameters: Dictionary mapping parameter names to type definitions.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        List of generated token IDs for the parameters.
    """
    generated: List[int] = []
    generated.extend(encode_cached(model, '{'))
    params = list(parameters.items())
    for i, (key, typedef) in enumerate(params):
        generated.extend(encode_cached(model, f'"{key}": '))
        value_ids = _generate_value_ids(model, input_ids + generated, typedef)
        generated.extend(value_ids)
        if i < len(params) - 1:
            generated.extend(encode_cached(model, ', '))
    generated.extend(encode_cached(model, '}'))
    return generated


def generate_function_call(model: Small_LLM_Model, request: Prompt,
                           function: FunctionDef,
                           visualize: bool = False) -> FunctionCall:
    """Generate a function call from a natural language prompt.

    Args:
        model: The LLM model instance.
        request: The user's prompt.
        function: The selected function definition.
        visualize: Whether to print each generated token to the terminal.

    Returns:
        A FunctionCall object containing the function name and arguments.
    """
    parameters_prompt = _build_parameters_prompt(request, function)
    input_ids = model.encode(parameters_prompt).tolist()[0]
    parameters_ids = _generate_parameters_ids(
        model, input_ids, function.parameters, visualize)
    parameters = json.loads(model.decode(parameters_ids))
    return FunctionCall(
        prompt=request.prompt, name=function.name, parameters=parameters)
