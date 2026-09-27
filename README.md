*This project was created as part of the 42 curriculum by hwakatsu.*

# Call Me Maybe

Call Me Maybe converts natural-language requests into structured function
calls with a local language model. Instead of accepting arbitrary model
output and repairing it afterward, the program restricts the tokens that can
be selected during function-name and argument generation.

The project demonstrates:

- token-constrained function selection
- typed argument generation for strings, numbers, booleans, objects, and arrays
- recursive function schemas with Pydantic input validation
- a custom tokenizer built from `vocab.json`
- integration tests against the real local model

## Technical Core

The pipeline performs two model-guided passes for each request.

### 1. Function selection

`src/selector.py` encodes every available function name as a candidate token
sequence. At each step, the model may select only a token that continues at
least one remaining candidate. A trailing newline distinguishes a complete
function name from a longer name that shares the same prefix.

### 2. Argument generation

`src/generator.py` uses the selected function's parameter definitions to
build a JSON object:

- object keys and JSON delimiters are inserted by the program
- booleans are selected from the token sequences for `true` and `false`
- number generation is limited to numeric characters and JSON terminators
- objects and arrays are generated recursively
- strings are generated until a token containing a closing quote is selected

The completed token sequence is decoded and parsed with `json.loads` before a
`FunctionCall` result is created.

## Design

| Component | Responsibility |
|---|---|
| `src/loader.py` | Loads and validates prompts, function definitions, and the model |
| `src/schema.py` | Defines Pydantic models for inputs and results |
| `src/selector.py` | Selects a function from constrained candidates |
| `src/generator.py` | Generates arguments according to the selected schema |
| `src/encode.py` | Encodes text using a greedy longest-match tokenizer |
| `src/decode.py` | Decodes token IDs and streams generated tokens |
| `src/utils.py` | Handles vocabulary loading, output, and shared errors |

Function definitions use a recursive `TypeDef`, allowing nested objects and
arrays without a separate implementation for every possible shape. Fixed
text such as function names and JSON syntax is encoded through an
`lru_cache`-backed encoder and reused across requests.

## Example

Input:

```text
What is the sum of 2 and 3?
```

Selected function and generated output:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

With `--visualize`, function-name and argument tokens are printed as they are
selected.

## Testing

The test suite contains two kinds of checks:

- **Real-model integration tests** exercise function selection and argument
  generation for numbers, strings, booleans, nested objects, and arrays.
- **Deterministic unit tests** cover shared-prefix function names, the numeric
  token filter, recursive Pydantic schemas, tokenizer behavior, and error
  handling without loading the model.

Input error cases cover missing files, invalid JSON, invalid function
definitions, empty prompt files, and an unknown model name.

```bash
make test
make lint
```

`make lint` runs `flake8` and `mypy` with type-checking options defined in the
Makefile.

## Limitations

- Numeric token filtering narrows the model's choices but does not implement
  the complete JSON number grammar as a state machine. `json.loads` is the
  final syntax check.
- String generation stops at the first generated quote and does not provide a
  complete JSON escaping grammar for arbitrary model output.
- Pydantic validates the prompt and function-definition structures. Generated
  parameters are guided by `FunctionDef`, but `FunctionCall.parameters` is a
  `Dict[str, Any]` rather than a dynamically generated Pydantic model.
- The custom tokenizer uses greedy longest matching over `vocab.json`, not the
  model's learned BPE merge order. It matches the SDK for the tested inputs but
  can differ on other text.
- The SDK exposes inference for one token sequence at a time, so this project
  does not implement batched generation.

## Requirements and Setup

- Python 3.10 or later
- [`uv`](https://docs.astral.sh/uv/)
- enough local memory and disk space for the selected Hugging Face model

```bash
make install
make run
```

The default model is `Qwen/Qwen3-0.6B`. The first run may retrieve model files
from Hugging Face; generation runs locally after the model is loaded.

Useful commands:

```bash
make run-visualize  # stream constrained generation
make debug          # run with pdb
make test           # run pytest
make lint           # run flake8 and mypy
make lint-strict    # run mypy --strict
make clean          # remove Python caches
make fclean         # also remove the virtual environment and generated output
```

The CLI accepts `--functions_definition`, `--input`, `--output`, `--model`,
and `--visualize` options.

## AI Assistance

AI was used as a design and review aid for debugging discussions, tokenizer
trade-offs, type-specific token masking, and documentation review. The final
implementation decisions and validation remained the author's responsibility.
