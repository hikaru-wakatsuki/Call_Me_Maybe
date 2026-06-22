import pytest
from typing import List, Dict, Tuple, Callable
from llm_sdk import Small_LLM_Model  # type: ignore
from src.loader import load_functions, load_prompts, load_model
from src.selector import build_functions_tokens, select_function
from src.generator import generate_function_call
from src.schema import Prompt, FunctionDef
from src.utils import load_vocab
from src.encode import build_cached_encoder, encode_custom
from src.decode import decode_custom


FUNCTIONS = "tests/data/functions.json"
INVALID = "tests/data/invalid.json"
BAD_SCHEMA = "tests/data/bad_schema.json"
EMPTY = "tests/data/empty.json"

Vocab = Tuple[Dict[str, int], Dict[int, str]]


@pytest.fixture(scope="session")
def model() -> Small_LLM_Model:
    return Small_LLM_Model()


@pytest.fixture(scope="session")
def functions() -> List[FunctionDef]:
    return load_functions(FUNCTIONS)


@pytest.fixture(scope="session")
def vocab(model: Small_LLM_Model) -> Vocab:
    return load_vocab(model)


@pytest.fixture(scope="session")
def encode_cached(vocab: Vocab) -> Callable[[str], List[int]]:
    token_to_id, _ = vocab
    return build_cached_encoder(token_to_id)


@pytest.fixture(scope="session")
def functions_tokens(
        functions: List[FunctionDef],
        encode_cached: Callable[[str], List[int]]) -> Dict[str, List[int]]:
    return build_functions_tokens(functions, encode_cached)


class TestNormalCases:
    def test_add_numbers(
            self, model: Small_LLM_Model, functions: List[FunctionDef],
            functions_tokens: Dict[str, List[int]], vocab: Vocab,
            encode_cached: Callable[[str], List[int]]) -> None:
        print()
        print("=== test_add_numbers ===")
        token_to_id, id_to_token = vocab
        prompt = Prompt(prompt="What is the sum of 2 and 3?")
        function = select_function(
            model, prompt, functions, functions_tokens,
            token_to_id, id_to_token)
        result = generate_function_call(
            model, prompt, function, token_to_id, id_to_token, encode_cached)
        print(f"prompt  : {prompt.prompt}")
        print("expected: fn_add_numbers({'a': 2, 'b': 3})")
        print(f"actual  : {result.name}({result.parameters})")
        assert result.name == "fn_add_numbers"
        assert result.parameters["a"] == 2
        assert result.parameters["b"] == 3

    def test_greet(
            self, model: Small_LLM_Model, functions: List[FunctionDef],
            functions_tokens: Dict[str, List[int]], vocab: Vocab,
            encode_cached: Callable[[str], List[int]]) -> None:
        print()
        print("=== test_greet ===")
        token_to_id, id_to_token = vocab
        prompt = Prompt(prompt="Greet John")
        function = select_function(
            model, prompt, functions, functions_tokens,
            token_to_id, id_to_token)
        result = generate_function_call(
            model, prompt, function, token_to_id, id_to_token, encode_cached)
        print(f"prompt  : {prompt.prompt}")
        print("expected: fn_greet({'name': 'John'})")
        print(f"actual  : {result.name}({result.parameters})")
        assert result.name == "fn_greet"
        assert result.parameters["name"] == "John"

    def test_reverse_string(
            self, model: Small_LLM_Model, functions: List[FunctionDef],
            functions_tokens: Dict[str, List[int]], vocab: Vocab,
            encode_cached: Callable[[str], List[int]]) -> None:
        print()
        print("=== test_reverse_string ===")
        token_to_id, id_to_token = vocab
        prompt = Prompt(prompt="Reverse the string 'hello'")
        function = select_function(
            model, prompt, functions, functions_tokens,
            token_to_id, id_to_token)
        result = generate_function_call(
            model, prompt, function, token_to_id, id_to_token, encode_cached)
        print(f"prompt  : {prompt.prompt}")
        print("expected: fn_reverse_string({'s': 'hello'})")
        print(f"actual  : {result.name}({result.parameters})")
        assert result.name == "fn_reverse_string"
        assert result.parameters["s"] == "hello"

    def test_set_active(
            self, model: Small_LLM_Model, functions: List[FunctionDef],
            functions_tokens: Dict[str, List[int]], vocab: Vocab,
            encode_cached: Callable[[str], List[int]]) -> None:
        print()
        print("=== test_set_active ===")
        token_to_id, id_to_token = vocab
        prompt = Prompt(prompt="Set is_active to true")
        function = select_function(
            model, prompt, functions, functions_tokens,
            token_to_id, id_to_token)
        result = generate_function_call(
            model, prompt, function, token_to_id, id_to_token, encode_cached)
        print(f"prompt  : {prompt.prompt}")
        print("expected: fn_set_active({'is_active': True})")
        print(f"actual  : {result.name}({result.parameters})")
        assert result.name == "fn_set_active"
        assert result.parameters["is_active"]

    def test_create_user(
            self, model: Small_LLM_Model, functions: List[FunctionDef],
            functions_tokens: Dict[str, List[int]], vocab: Vocab,
            encode_cached: Callable[[str], List[int]]) -> None:
        print()
        print("=== test_create_user ===")
        token_to_id, id_to_token = vocab
        prompt = Prompt(prompt="Create a user with name Alice and age 30")
        function = select_function(
            model, prompt, functions, functions_tokens,
            token_to_id, id_to_token)
        result = generate_function_call(
            model, prompt, function, token_to_id, id_to_token, encode_cached)
        print(f"prompt  : {prompt.prompt}")
        print("expected: fn_create_user("
              "{'user': {'name': 'Alice', 'age': 30}})")
        print(f"actual  : {result.name}({result.parameters})")
        assert result.name == "fn_create_user"
        assert result.parameters["user"]["name"] == "Alice"
        assert result.parameters["user"]["age"] == 30

    def test_tag_item(
            self, model: Small_LLM_Model, functions: List[FunctionDef],
            functions_tokens: Dict[str, List[int]], vocab: Vocab,
            encode_cached: Callable[[str], List[int]]) -> None:
        print()
        print("=== test_tag_item ===")
        token_to_id, id_to_token = vocab
        prompt = Prompt(prompt="Tag the item with python, ai and llm")
        function = select_function(
            model, prompt, functions, functions_tokens,
            token_to_id, id_to_token)
        result = generate_function_call(
            model, prompt, function, token_to_id, id_to_token, encode_cached)
        print(f"prompt  : {prompt.prompt}")
        print("expected: fn_tag_item({'tags': ['python', 'ai', 'llm']})")
        print(f"actual  : {result.name}({result.parameters})")
        assert result.name == "fn_tag_item"
        assert "python" in result.parameters["tags"]
        assert "ai" in result.parameters["tags"]
        assert "llm" in result.parameters["tags"]


class TestTokenizer:
    def test_encode_matches_sdk(
            self, model: Small_LLM_Model, vocab: Vocab) -> None:
        print()
        print("=== test_encode_matches_sdk ===")
        token_to_id, _ = vocab
        text = "What is the sum of 2 and 3?"
        sdk_ids = model.encode(text).tolist()[0]
        custom_ids = encode_custom(text, token_to_id)
        print(f"prompt  : {text!r}")
        print(f"expected: {sdk_ids}")
        print(f"actual  : {custom_ids}")
        assert custom_ids == sdk_ids

    def test_decode_matches_sdk(
            self, model: Small_LLM_Model, vocab: Vocab) -> None:
        print()
        print("=== test_decode_matches_sdk ===")
        _, id_to_token = vocab
        text = "What is the sum of 2 and 3?"
        ids = model.encode(text).tolist()[0]
        sdk_text = model.decode(ids)
        custom_text = decode_custom(ids, id_to_token)
        print(f"prompt  : {ids}")
        print(f"expected: {sdk_text!r}")
        print(f"actual  : {custom_text!r}")
        assert custom_text == sdk_text


class TestErrorCases:
    def test_file_not_found(self) -> None:
        print()
        print("=== test_file_not_found ===")
        print("prompt  : load_functions('nonexistent.json')")
        print("expected: Error: File not found: nonexistent.json")
        with pytest.raises(SystemExit):
            print("actual  : ", end="")
            load_functions("nonexistent.json")

    def test_invalid_json(self) -> None:
        print()
        print("=== test_invalid_json ===")
        print(f"prompt  : load_functions('{INVALID}')")
        print(f"expected: Error: Invalid JSON in {INVALID}")
        with pytest.raises(SystemExit):
            print("actual  : ", end="")
            load_functions(INVALID)

    def test_bad_schema(self) -> None:
        print()
        print("=== test_bad_schema ===")
        print(f"prompt  : load_functions('{BAD_SCHEMA}')")
        print(f"expected: Error: Invalid data in {BAD_SCHEMA}")
        with pytest.raises(SystemExit):
            print("actual  : ", end="")
            load_functions(BAD_SCHEMA)

    def test_prompts_empty(self) -> None:
        print()
        print("=== test_empty ===")
        prompts = load_prompts(EMPTY)
        print(f"prompt  : load_prompts('{EMPTY}')")
        print("expected: []")
        print(f"actual  : {prompts}")
        assert prompts == []

    def test_model_not_found(self) -> None:
        print()
        print("=== test_model_not_found ===")
        print("prompt  : load_model('nonexistent/model-xyz')")
        print("expected: Error: Failed to load model 'nonexistent/model-xyz'")
        with pytest.raises(SystemExit):
            print("actual  : ", end="")
            load_model("nonexistent/model-xyz")
