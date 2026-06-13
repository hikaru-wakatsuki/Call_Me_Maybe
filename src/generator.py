from llm_sdk import Small_LLM_Model  # type: ignore
from typing import List, Dict
from .schema import FunctionDef, Prompt, TypeDef, FunctionCall
import json


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


def _generate_primitive(
        model: Small_LLM_Model, input_ids: List[int]) -> List[int]:
    """Generate token IDs for a primitive value (number, boolean, etc.).

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.

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
        generated.append(next_id)
    return generated


def _generate_string(
        model: Small_LLM_Model, input_ids: List[int]) -> List[int]:
    """Generate token IDs for a string value, wrapped in double quotes.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.

    Returns:
        List of generated token IDs for the string value.
    """
    generated: List[int] = []
    generated.extend(model.encode('"').tolist()[0])
    for _ in range(MAX_STEPS):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = logits.index(max(logits))
        token = model.decode([next_id])
        if '"' in token:
            break
        generated.append(next_id)
    generated.extend(model.encode('"').tolist()[0])
    return generated


def _generate_array(model: Small_LLM_Model, input_ids: List[int],
                    typedef: TypeDef) -> List[int]:
    """Generate token IDs for an array value.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        typedef: The type definition of the array elements.

    Returns:
        List of generated token IDs for the array value.
    """
    generated: List[int] = []
    generated.extend(model.encode('[').tolist()[0])
    for _ in range(MAX_STEPS):
        value_ids = _generate_value(model, input_ids + generated, typedef)
        generated.extend(value_ids)
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = logits.index(max(logits))
        token = model.decode([next_id])
        if ']' in token:
            break
        generated.extend(model.encode(',').tolist()[0])
    generated.extend(model.encode(']').tolist()[0])
    return generated


def _generate_value(model: Small_LLM_Model, input_ids: List[int],
                    typedef: TypeDef) -> List[int]:
    """Generate value tokens for a function argument using constrained
       decoding.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        typedef: The type definition of the argument.

    Returns:
        List of generated token IDs for the value.
    """
    if typedef.properties is not None:
        return _generate_parameters(model, input_ids, typedef.properties)
    elif typedef.items is not None:
        return _generate_array(model, input_ids, typedef.items)
    elif typedef.type in ("string", "str"):
        return _generate_string(model, input_ids)
    else:
        return _generate_primitive(model, input_ids)


def _generate_parameters(model: Small_LLM_Model, input_ids: List[int],
                         parameters:  Dict[str, TypeDef]) -> List[int]:
    """Generate arguments for a function call using constrained decoding.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the prompt.
        parameters: Dictionary mapping parameter names to type definitions.

    Returns:
        List of generated token IDs for the parameters.
    """
    generated: List[int] = []
    generated.extend(model.encode("{").tolist()[0])
    params = list(parameters.items())
    for i, (key, typedef) in enumerate(params):
        generated.extend(model.encode(f'"{key}": ').tolist()[0])
        value_ids = _generate_value(model, input_ids + generated, typedef)
        generated.extend(value_ids)
        if i < len(params) - 1:
            generated.extend(model.encode(", ").tolist()[0])
    generated.extend(model.encode("}").tolist()[0])
    return generated


def generate_function_call(model: Small_LLM_Model, request: Prompt,
                           function: FunctionDef) -> FunctionCall:
    """Generate a function call from a natural language prompt.

    Args:
        model: The LLM model instance.
        request: The user's prompt.
        function: The selected function definition.

    Returns:
        A FunctionCall object containing the function name and arguments.
    """
    parameters_prompt = _build_parameters_prompt(request, function)
    input_ids = model.encode(parameters_prompt).tolist()[0]
    parameters_ids = _generate_parameters(
        model, input_ids, function.parameters)
    parameters = json.loads(model.decode(parameters_ids))
    return FunctionCall(
        prompt=request.prompt, name=function.name, parameters=parameters)
