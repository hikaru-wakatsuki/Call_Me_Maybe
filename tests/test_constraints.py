from typing import Any, Dict, List, cast

import pytest
from pydantic import ValidationError

from src.decode import decode_custom
from src.encode import encode_custom
from src.schema import FunctionDef, Prompt, TypeDef
from src.selector import _generate_function_ids, build_functions_tokens
from src.utils import build_number_token_ids, handle_error


class FakeModel:
    """Return deterministic logits for a requested token sequence."""

    def __init__(self, prefix_length: int, generated: List[int]) -> None:
        self.prefix_length = prefix_length
        self.generated = generated

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:
        step = len(input_ids) - self.prefix_length
        logits = [0.0] * 16
        logits[self.generated[step]] = 1.0
        return logits


@pytest.mark.parametrize(
    ("generated", "expected"),
    [
        ([1, 2, 3, 9], [1, 2, 3, 9]),
        ([1, 2, 9], [1, 2, 9]),
    ],
)
def test_function_selection_handles_shared_prefixes(
        generated: List[int], expected: List[int]) -> None:
    function_tokens = {
        "fn_add": [1, 2, 9],
        "fn_add_plus": [1, 2, 3, 9],
    }
    id_to_token = {1: "fn", 2: "_add", 3: "_plus", 9: "\n"}
    model = FakeModel(prefix_length=1, generated=generated)

    result = _generate_function_ids(
        cast(Any, model), [15], function_tokens, id_to_token)

    assert result == expected


def test_function_tokens_include_terminating_newline() -> None:
    function = FunctionDef(
        name="fn_greet",
        description="Greet a user.",
        parameters={"name": TypeDef(type="string")},
        returns=TypeDef(type="string"),
    )
    encoded_inputs: List[str] = []

    def encode(text: str) -> List[int]:
        encoded_inputs.append(text)
        return list(range(len(text)))

    result = build_functions_tokens([function], encode)

    assert encoded_inputs == ["fn_greet\n"]
    assert result["fn_greet"] == list(range(len("fn_greet\n")))


def test_number_token_filter_rejects_non_numeric_characters() -> None:
    id_to_token = {
        0: "123",
        1: "-4.5",
        2: ",",
        3: "abc",
        4: "1e3",
        5: "",
    }

    assert build_number_token_ids(id_to_token) == [0, 1, 2]


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "hello", "unexpected": True},
    ],
)
def test_prompt_rejects_invalid_input(payload: Dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Prompt(**payload)


def test_recursive_type_definition_accepts_nested_schema() -> None:
    schema = TypeDef(
        type="object",
        properties={
            "user": TypeDef(
                type="object",
                properties={
                    "name": TypeDef(type="string"),
                    "tags": TypeDef(
                        type="array",
                        items=TypeDef(type="string"),
                    ),
                },
            ),
        },
    )

    assert schema.properties is not None
    assert schema.properties["user"].properties is not None
    assert schema.properties["user"].properties["tags"].items is not None


def test_custom_tokenizer_round_trip_with_space_and_newline() -> None:
    token_to_id = {
        "Hello": 0,
        "Ġworld": 1,
        "Ċ": 2,
        "42": 3,
    }
    id_to_token = {token_id: token for token, token_id in token_to_id.items()}

    encoded = encode_custom("Hello world\n42", token_to_id)

    assert encoded == [0, 1, 2, 3]
    assert decode_custom(encoded, id_to_token) == "Hello world\n42"


def test_handle_error_reports_message_and_exit_status(capsys: Any) -> None:
    with pytest.raises(SystemExit) as error:
        handle_error("example error")

    assert error.value.code == 1
    assert capsys.readouterr().out == "example error\n"
