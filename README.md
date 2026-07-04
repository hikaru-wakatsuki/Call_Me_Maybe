*This project has been created as part of the 42 curriculum by hwakatsu.*

# Call Me Maybe

## Description

This project converts natural language prompts into structured function
calls (JSON) using a local LLM (`Qwen/Qwen3-0.6B` by default). Given a list
of available functions and a set of prompts, the program decides which
function each prompt is asking for and produces a fully valid JSON call with
correctly typed arguments.

The core technical challenge is guaranteeing that every output is valid JSON
and conforms to the target function's schema, even though the underlying
model has no built-in notion of JSON syntax or argument types. This is
solved with **constrained decoding**: at every generation step the model is
only allowed to choose from a restricted set of tokens that keep the output
structurally and semantically valid. As a result, the program never produces
malformed JSON and never needs to retry or repair its output.

## Instructions

The project uses [`uv`](https://docs.astral.sh/uv/) for dependency
management and a `Makefile` for all common tasks.

```bash
# Install dependencies (including the local llm_sdk)
make install

# Run with default model and input/output paths
make run

# Watch the constrained decoding process token by token
make run-visualize

# Run the program under the Python debugger
make debug

# Run the linters (flake8 + mypy, default rules)
make lint

# Run the linters with mypy --strict
make lint-strict

# Run the test suite
make test

# Remove Python caches and bytecode
make clean

# Remove the virtual environment and generated output (full cleanup)
make fclean
```

`make run` and `make run-visualize` both run `python -m src`, which also
accepts the following CLI options directly if you need to override a
default (e.g. `uv run python -m src --model Qwen/Qwen3-0.6B`):

- `--functions_definition` — path to the function definitions JSON
  (default `data/input/functions_definition.json`).
- `--input` — path to the prompts JSON
  (default `data/input/function_calling_tests.json`).
- `--output` — path to the results JSON
  (default `data/output/function_calling_results.json`).
- `--model` — Hugging Face model identifier (default `Qwen/Qwen3-0.6B`).
- `--visualize` — stream each generated token to the terminal in real time.

## Algorithm Explanation

### The generation pipeline

For each prompt, the program runs two constrained-decoding passes:

1. **Function selection** (`src/selector.py`) — the model is shown the list
   of available functions and the user's prompt, then generates the function
   name one token at a time. At each step, only tokens that keep the partial
   output a valid prefix of one of the candidate function names are
   considered.
2. **Argument generation** (`src/generator.py`) — once a function is
   selected, the model generates its arguments as JSON, one token at a time.
   The generator tracks the expected JSON structure (`{`, `}`, `[`, `]`,
   `,`, key names) and each parameter's declared type to constrain every
   step.

### Constrained decoding in practice

At each generation step:

1. The model produces logits for every token in the vocabulary
   (`get_logits_from_input_ids`).
2. The program determines which tokens would keep the partial output valid
   (a still-unfinished function name, a still-open string, a value of the
   correct type, etc.).
3. Among the valid candidates, the token with the highest logit is chosen.
4. The chosen token is appended, and the loop repeats until the function
   name or the full JSON argument object is complete.

Because the set of allowed tokens is restricted *before* the choice is made,
every generated token is guaranteed to keep the output valid. This produces
100% valid JSON without any retry-or-repair step. Generation is fully
deterministic (the highest-logit valid token is always chosen), so the same
prompt and model always yield the same result.

### Type-specific masking

The masking strategy differs per JSON value type, but always follows the
same principle — restrict the candidate pool *before* picking the
highest-logit token, never after:

- **`object` / `array`** — only the fixed structural tokens (`{`, `}`, `[`,
  `]`, `,`, key names) are inserted directly; the model is never asked to
  produce them.
- **`string`** — the model generates freely until a token containing a
  closing `"` is produced; any text preceding the quote within that same
  token is still kept.
- **`number`** — at each step, the candidate pool is restricted to
  vocabulary tokens consisting only of digits, `.`, `-`, or a JSON
  terminator (`,`, `}`, `]`), so the model cannot emit a non-numeric token.
- **`boolean`** — the candidate pool is restricted to exactly two full
  token sequences, `"true"` and `"false"`; at each step only sequences
  still matching what has been generated so far remain eligible, and the
  model picks between them by logit, the same mechanism used for function
  name selection.

## Design Decisions

- **Pydantic for all data models** (`src/schema.py`): prompts, function
  definitions, type definitions, and results are validated pydantic models
  with `extra='forbid'`, so malformed input is rejected early with a clear
  message instead of failing deep inside the generation logic.
- **Recursive `TypeDef`**: nested object and array parameters are modeled
  with a single recursive type (`properties` for objects, `items` for
  arrays), so arbitrarily nested arguments are supported without
  special-casing each shape.
- **Separation of concerns**: `loader.py` reads and validates files,
  `selector.py` picks a function, `generator.py` fills its arguments, and
  `encode.py` / `decode.py` handle tokenization. Each file has a small,
  testable responsibility.
- **Custom tokenizer in the main path**: `selector.py` and `generator.py`
  call only `encode_custom` / `decode_custom` / `get_logits_from_input_ids`.
  The SDK's `model.encode` / `model.decode` are never used in the main code.
- **Caching scoped to fixed text only**: the cached encoder
  (`build_cached_encoder`) is used only for strings that repeat across
  prompts — function names and JSON structural tokens. The natural-language
  prompt is never cached, since it differs on every call and would never
  produce a cache hit.
- **Function names are tokenized with a trailing newline**: each candidate
  function name is encoded as `name + "\n"` rather than just `name`. This
  guarantees no candidate's token sequence is a prefix of another's, so the
  selection loop can always tell whether the model intends to stop at the
  current name or continue into a longer one (see Challenges Faced).
- **Type-aware masking for numbers and booleans**: `number` arguments
  restrict the candidate pool to digit/`.`/`-`/terminator tokens, and
  `boolean` arguments choose between exactly two full candidates (`"true"`,
  `"false"`). Both follow the same logit-restriction principle already used
  for JSON structural tokens and function names, rather than relying on the
  model to spontaneously produce a correctly-typed value.

## Performance Analysis

- **Accuracy**: constrained decoding guarantees structurally and
  schema-valid JSON on every prompt. The custom tokenizer was verified to
  produce token sequences identical to the SDK's `encode`/`decode` for every
  prompt in the test set.
- **Speed**: the cached encoder tokenizes each fixed token (function names,
  JSON syntax) once instead of once per prompt. The greedy encoder works on
  word-level chunks, so the per-step match cost is bounded by the length of
  a single word rather than the whole prompt.
- **Reliability**: every generation loop is bounded by `MAX_STEPS`, so a
  misbehaving model can never hang the program; and every error path exits
  cleanly through `handle_error` rather than raising an unhandled exception.

## Challenges Faced

- **Token-length mismatch between candidate function names**: the
  function-selection loop originally assumed every candidate name had the
  same number of tokens, causing an `IndexError` once names of different
  token lengths were compared at the same step. Fixed by only keeping
  candidates whose token sequence still matches everything generated so
  far (`tokens[:len(generated)] == generated`), rather than only checking
  remaining length.
- **One function name being a prefix of another**: if two function names
  share a token prefix (e.g. `fn_add` and `fn_add_plus`), the original loop
  would stop as soon as `generated` matched the shorter name exactly, even
  if the model intended to continue into the longer one — the shorter name
  could never lose that comparison once reached. Fixed by tokenizing each
  function name with a trailing newline, so no name's tokens are ever a
  prefix of another's; the model can then genuinely choose between
  "stop here" and "continue" at every shared step.
- **Number and boolean values were initially unmasked**: an early version
  generated `number` and `boolean` values by simply picking the
  highest-logit token with no restriction, relying on the model to
  spontaneously produce a valid value — exactly the approach the subject
  warns against. This could silently produce a non-numeric token or a `1`
  instead of `true`. Fixed by adding the same logit-restriction principle
  used elsewhere: a precomputed digit/`.`/`-` mask for numbers, and a
  two-candidate (`"true"`/`"false"`) restriction for booleans.
- **Byte-level markers beyond spaces**: the subject example only documents
  the space marker (`Ġ`), but the program's internally built prompts contain
  newlines, so the custom tokenizer also had to handle the newline marker
  (`Ċ`).
- **Streaming-visualization timing**: token-by-token printing must flush
  manually (`flush=True`) and must only print tokens that are actually kept.
  Printing a token that is later discarded (e.g. a JSON delimiter detected
  one step too early) produced duplicated or missing characters until the
  print position was aligned with the append logic. The trailing newline
  added to terminate function-name generation also had to be stripped from
  the visualized output specifically, to avoid an extra blank line between
  the selected function name and the next printed line.

## Testing Strategy

`tests/test_integration.py` (run with `make test`) is an integration suite
that exercises the real model end to end:

- **Normal cases** — one test per parameter type (number, string, boolean,
  nested object, array), asserting the correct function and arguments.
- **Tokenizer equivalence** — `encode_custom` / `decode_custom` are compared
  directly against the SDK's `encode` / `decode` on the same input.
- **Error cases** — missing files, invalid JSON, schema violations, empty
  prompt lists, and unknown model names all assert a clean exit via
  `handle_error` rather than an unhandled exception.

The code also passes `flake8` and `mypy` (with `--disallow-untyped-defs`,
`--check-untyped-defs`, `--warn-return-any`, `--warn-unused-ignores`,
`--ignore-missing-imports`) via `make lint`.

## Example Usage

Input prompt:

```
"What is the sum of 2 and 3?"
```

With `--visualize`, the program streams the decoding process:

```
============================================================
Prompt: What is the sum of 2 and 3?
Selecting function: fn_add_numbers
Generating arguments: {"a": 2, "b": 3}
```

Resulting entry written to the output JSON
(`data/output/function_calling_results.json`):

```json
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2, "b": 3}
}
```

## Bonus Features

| Feature | Status |
|---|---|
| Multiple LLM model support (`--model`) | Implemented |
| Nested complex function arguments | Implemented |
| Comprehensive test suite | Implemented |
| Visualization of the generation process (`--visualize`) | Implemented |
| Advanced error recovery mechanisms | Implemented |
| Performance optimization — caching | Implemented |
| Performance optimization — batching | Not implemented (see below) |
| Tokenizer reimplementation (`encode_custom` / `decode_custom`) | Implemented |
| Public `encode` / `decode` implementation | Implemented |
| Integration of custom tokenizer with constrained decoding | Implemented |

**Custom tokenizer (greedy longest-match):** `encode_custom` reimplements
tokenization using only `vocab.json` (`get_path_to_vocab_file`). Spaces and
newlines are replaced with their byte-level markers (`Ġ`, `Ċ`), the text is
split into word-level chunks (BPE never merges across a word boundary, so
this is a safe, non-lossy optimization), and within each chunk the longest
matching vocabulary entry is consumed repeatedly. Known limitation: because
the learned merge order (`merges.txt`) is intentionally not used, greedy
matching can occasionally diverge from the original BPE tokenizer on unusual
or slang words (e.g. "shrek") that do not appear in the test data. This is an
accepted trade-off — the subject only requires `vocab.json`.

**Why batching was not implemented:** the SDK's only inference entry point,
`get_logits_from_input_ids(input_ids: list[int])`, accepts exactly one token
sequence at a time, with no documented way to pass a batch dimension. True
batching (stacking several prompts into one tensor for a single forward
pass) is therefore impossible through the public API.

## Resources

- The 42 subject PDF for this project (assignment specification).
- The `llm_sdk` source code, for the exact behavior of `encode`, `decode`,
  and `get_logits_from_input_ids`.
- Hugging Face documentation on tokenizers and byte-level BPE, for the
  meaning of the `Ġ` / `Ċ` markers and how byte-level encoding works.

### Use of AI

AI assistance (Claude) was used as a design-and-review partner throughout
the project. Specifically:

- **Design discussion**: weighing where to place the token cache, how to
  structure the custom tokenizer (greedy longest-match vs. full BPE), how
  to split responsibilities across files, and how to mask candidate tokens
  per JSON value type (string/number/boolean/object/array).
- **Debugging**: diagnosing the function-name `IndexError`, the
  prefix-collision bug between candidate function names, the missing
  newline marker in the custom tokenizer, the unmasked number/boolean
  generation gap, and the token-printing misalignment in the
  visualization.
- **Code review**: checking type hints, docstrings, and `flake8`/`mypy`
  compliance, cross-checking the implementation against the subject's
  requirements, and reviewing the wording of this README.

All code was reviewed, understood, and validated by the author before being
committed.
