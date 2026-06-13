from llm_sdk import Small_LLM_Model  # type: ignore
from typing import List, Dict
from .schema import FunctionDef, Prompt, TypeDef, FunctionCall
import json
from .selector import build_selection_prompt, select_function


MAX_STEPS = 100


def build_parameters_prompt(request: Prompt, function: FunctionDef) -> str:
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


def generate_tokens(model: Small_LLM_Model, input_ids: List[int]) -> List[int]:
    generated: List[int] = []
    for _ in range(MAX_STEPS):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = logits.index(max(logits))
        token = model.decode([next_id])
        if ',' in token or '}' in token or ']' in token:
            break
        generated.append(next_id)
    return generated


def generate_string(model: Small_LLM_Model, input_ids: List[int]) -> List[int]:
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


def generate_array(model: Small_LLM_Model, input_ids: List[int],
                   typedef: TypeDef) -> List[int]:
    generated: List[int] = []
    generated.extend(model.encode('[').tolist()[0])
    for _ in range(MAX_STEPS):
        value_ids = generate_value(model, input_ids + generated, typedef)
        generated.extend(value_ids)
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = logits.index(max(logits))
        token = model.decode([next_id])
        if ']' in token:
            break
        generated.extend(model.encode(',').tolist()[0])
    generated.extend(model.encode(']').tolist()[0])
    return generated


def generate_value(model: Small_LLM_Model, input_ids: List[int],
                   typedef: TypeDef) -> List[int]:
    """Generate value tokens for a function argument using constrained
       decoding.

    Args:
        model: The LLM model instance.
        typedef: The type definition of the argument.
        input_ids: The input token IDs including the prompt.

    Returns:
        List of generated token IDs for the value.
    """
    if typedef.properties is not None:
        return generate_parameters(model, input_ids, typedef.properties)
    elif typedef.items is not None:
        return generate_array(model, input_ids, typedef.items)
    elif typedef.type in ("string", "str"):
        return generate_string(model, input_ids)
    else:
        return generate_tokens(model, input_ids)


def generate_parameters(model: Small_LLM_Model, input_ids: List[int],
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
        value_ids = generate_value(model, input_ids + generated, typedef)
        generated.extend(value_ids)
        if i < len(params) - 1:
            generated.extend(model.encode(", ").tolist()[0])
    generated.extend(model.encode("}").tolist()[0])
    return generated


def generate(
        model: Small_LLM_Model, request: Prompt, functions: List[FunctionDef],
        functions_tokens: Dict[str, List[int]]) -> FunctionCall:
    """Generate a function call from a natural language prompt.

    Args:
        model: The LLM model instance.
        request: The user's prompt.
        functions: List of available function definitions.
        functions_tokens: Dictionary mapping function names to token IDs.

    Returns:
        A FunctionCall object containing the function name and arguments.
    """
    selection_prompt = build_selection_prompt(request, functions)
    selection_ids = model.encode(selection_prompt).tolist()[0]
    function = select_function(
        model, selection_ids, functions_tokens, functions)
    argument_prompt = build_parameters_prompt(request, function)
    argument_ids = model.encode(argument_prompt).tolist()[0]
    parameters_ids = generate_parameters(
        model, argument_ids, function.parameters)
    parameters = json.loads(model.decode(parameters_ids))
    return FunctionCall(
        prompt=request.prompt, name=function.name, parameters=parameters)
