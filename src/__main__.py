import argparse
from .loader import load_functions, load_prompts, load_model
from .selector import select_function
from .generator import generate_function_call
from .utils import write_output


def main() -> None:
    """Entry point for the function calling tool.

    Parses command line arguments, loads input files, runs the generation
    pipeline for each prompt, and writes results to the output file.
    """
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--functions_definition',
        default='data/input/functions_definition.json')
    parse.add_argument(
        '--input', default='data/input/function_calling_tests.json')
    parse.add_argument('--output', default='data/output/function_calls.json')
    parse.add_argument('--model', default='Qwen/Qwen3-0.6B')
    args = parse.parse_args()

    model = load_model(args.model)
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)

    results = []
    for prompt in prompts:
        function = select_function(model, prompt, functions)
        result = generate_function_call(model, prompt, function)
        results.append(result)

    write_output(args.output, results)


if __name__ == '__main__':
    main()
