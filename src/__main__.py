import argparse


def main() -> None:
    parse = argparse.ArgumentParser()
    parse.add_argument(
        '--functions_definition',
        default='data/input/functions_definition.json')
    parse.add_argument(
        '--input', default='data/input/function_calling_tests.json')
    parse.add_argument('--output', default='data/output/function_calls.json')
    args = parse.parse_args()


if __name__ == '__main__':
    main()
