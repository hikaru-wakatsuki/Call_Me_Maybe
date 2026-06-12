from llm_sdk import Small_LLM_Model
from typing import List, Dict
from .schema import FunctionDef, Prompt, TypeDef, FunctionCall
import json


MAX_STEPS = 100


def build_functions_tokens(
        model: Small_LLM_Model,
        functions: List[FunctionDef]) -> Dict[str, List[int]]:
    """Build a mapping of function names to their token IDs.

    Args:
        model: The LLM model instance.
        functions: List of function definitions.

    Returns:
        Dictionary mapping function names to lists of token IDs.
    """
    function_tokens = {}
    for function in functions:
        ids = model.encode(function.name).tolist()[0]
        function_tokens[function.name] = ids
    return function_tokens


def build_selection_prompt(
        request: Prompt, functions: List[FunctionDef]) -> str:
    """Build a prompt for function selection.

    Args:
        request: The user's prompt.
        functions: List of available function definitions.

    Returns:
        A prompt string for function selection.
    """
    prompt = "Available functions:"
    for function in functions:
        params = ", ".join(
            f"{k}: {v.type}" for k, v in function.parameters.items())
        prompt += f"\n- {function.name}({params}): {function.description}"
    prompt += f"\nUser request: {request.prompt}"
    prompt += "\nFunction name:"
    return prompt


def select_function(model: Small_LLM_Model, prompt: str,
                    function_tokens: Dict[str, List[int]],
                    functions: List[FunctionDef]) -> FunctionDef:
    """Select the appropriate function using constrained decoding.

    Args:
        model: The LLM model instance.
        prompt: The prompt string for function selection.
        functions_tokens: Dictionary mapping function names to token IDs.
        functions: List of available function definitions.

    Returns:
        The selected function definition.
    """
    input_ids = model.encode(prompt).tolist()[0]
    generated = []
    while True:
        logits = model.get_logits_from_input_ids(input_ids + generated)
        allowed = []
        for tokens in function_tokens.values():
            allowed.append(tokens[len(generated)])
        next_id = max(allowed, key=lambda i: logits[i])
        generated.append(next_id)
        for name, tokens in function_tokens.items():
            if tokens == generated:
                return next(f for f in functions if f.name == name)


def build_argument_prompt(request: Prompt, function: FunctionDef) -> str:
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


def generate_value(model: Small_LLM_Model,
                   typedef: TypeDef, input_ids: List[int]) -> List[int]:
    """Generate value tokens for a function argument using constrained
       decoding.

    Args:
        model: The LLM model instance.
        typedef: The type definition of the argument.
        input_ids: The input token IDs including the prompt.

    Returns:
        List of generated token IDs for the value.
    """
    generated = []
    is_string = typedef.type in ("string", "str")
    if is_string:
        generated.extend(model.encode('"').tolist()[0])
    for _ in range(MAX_STEPS):
        logits = model.get_logits_from_input_ids(input_ids + generated)
        next_id = logits.index(max(logits))
        token = model.decode([next_id])

        if is_string:
            if '"' in token:
                break
        else:
            if ',' in token or '}' in token:
                break

        generated.append(next_id)
    if is_string:
        generated.extend(model.encode('"').tolist()[0])
    return generated


def generate_argument(model: Small_LLM_Model, prompt: str,
                      function: FunctionDef) -> Dict[str, float | str | bool]:
    """Generate arguments for a function call using constrained decoding.

    Args:
        model: The LLM model instance.
        prompt: The prompt string for argument generation.
        function: The selected function definition.

    Returns:
        Dictionary mapping argument names to their values.
    """
    input_ids = model.encode(prompt).tolist()[0]
    generated = []
    generated.extend(model.encode("{").tolist()[0])
    params = list(function.parameters.items())
    for i, (key, typedef) in enumerate(params):
        generated.extend(model.encode(f'"{key}": ').tolist()[0])
        value_ids = generate_value(model, typedef, input_ids + generated)
        generated.extend(value_ids)
        if i < len(params) - 1:
            generated.extend(model.encode(", ").tolist()[0])
    generated.extend(model.encode("}").tolist()[0])
    result = model.decode(generated)
    return json.loads(result)


def generate(
        model: Small_LLM_Model, request: Prompt, functions: List[FunctionDef],
        functions_tokens: Dict[str, List[int]]) -> FunctionCall:
    selection_prompt = build_selection_prompt(request, functions)
    """Generate a function call from a natural language prompt.

    Args:
        model: The LLM model instance.
        request: The user's prompt.
        functions: List of available function definitions.
        functions_tokens: Dictionary mapping function names to token IDs.

    Returns:
        A FunctionCall object containing the function name and arguments.
    """
    function = select_function(
        model, selection_prompt, functions_tokens, functions)
    argument_prompt = build_argument_prompt(request, function)
    parameters = generate_argument(model, argument_prompt, function)
    return FunctionCall(
        prompt=request.prompt, name=function.name, parameters=parameters)
