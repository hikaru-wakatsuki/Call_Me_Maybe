from llm_sdk import Small_LLM_Model  # type: ignore
from typing import List, Dict
from .schema import FunctionDef, Prompt
from .utils import encode_cached


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
        ids = encode_cached(model, function.name)
        function_tokens[function.name] = ids
    return function_tokens


def _build_selection_prompt(
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


def _generate_function_ids(model: Small_LLM_Model, input_ids: List[int],
                           function_tokens: Dict[str, List[int]]) -> List[int]:
    """Select the appropriate function using constrained decoding.

    Args:
        model: The LLM model instance.
        input_ids: The input token IDs including the selection prompt.
        function_tokens: Dictionary mapping function names to token IDs.

    Returns:
        List of token IDs representing the selected function name.
    """
    generated: List[int] = []
    while True:
        logits = model.get_logits_from_input_ids(input_ids + generated)
        allowed = []
        for tokens in function_tokens.values():
            allowed.append(tokens[len(generated)])
        next_id = max(allowed, key=lambda i: logits[i])
        generated.append(next_id)
        for _, tokens in function_tokens.items():
            if tokens == generated:
                return generated


def select_function(model: Small_LLM_Model, request: Prompt,
                    functions: List[FunctionDef],
                    functions_tokens: Dict[str, List[int]]) -> FunctionDef:
    """Select the appropriate function for a given request.

    Args:
        model: The LLM model instance.
        request: The user's prompt.
        functions: List of available function definitions.
        functions_tokens: Mapping of function names to their token IDs,
            precomputed once and reused across prompts.

    Returns:
        The selected function definition.
    """
    selection_prompt = _build_selection_prompt(request, functions)
    selection_ids = model.encode(selection_prompt).tolist()[0]
    function_ids = _generate_function_ids(
        model, selection_ids, functions_tokens)
    function_name = model.decode(function_ids)
    return next(f for f in functions if f.name == function_name)
