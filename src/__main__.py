import argparse
from .loader import load_functions, load_prompts, load_model
from .selector import build_functions_tokens, select_function
from .generator import generate_function_call
from .utils import write_output, load_vocab, build_number_token_ids
from .encode import build_cached_encoder


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--functions_definition',
        default='data/input/functions_definition.json')
    parse.add_argument(
        '--input', default='data/input/function_calling_tests.json')
    parse.add_argument(
        '--output', default='data/output/function_calling_results.json')
    parse.add_argument('--model', default='Qwen/Qwen3-0.6B')
    parse.add_argument('--visualize', action='store_true')
    return parse.parse_args()


def main() -> None:
    """Entry point for the function calling tool.

    Parses command line arguments, loads input files, runs the generation
    pipeline for each prompt, and writes results to the output file.
    """
    args = _parse_args()
    model = load_model(args.model)
    token_to_id, id_to_token = load_vocab(model)
    number_token_ids = build_number_token_ids(id_to_token)
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)
    encode_cached = build_cached_encoder(token_to_id)
    functions_tokens = build_functions_tokens(functions, encode_cached)
    results = []
    for prompt in prompts:
        if args.visualize:
            print("=" * 60)
            print(f"Prompt: {prompt.prompt}")
        function = select_function(
            model, prompt, functions, functions_tokens,
            token_to_id, id_to_token, args.visualize)
        result = generate_function_call(
            model, prompt, function, token_to_id, id_to_token,
            number_token_ids, encode_cached, args.visualize)
        results.append(result)

    write_output(args.output, results)


if __name__ == '__main__':
    main()
