from pydantic import BaseModel, ConfigDict, Field
from typing import Dict, Any


class Prompt(BaseModel):
    """A user prompt for function calling.

    Attributes:
        prompt: The natural language request string.
    """
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1)


class TypeDef(BaseModel):
    """Type definition for a function parameter.

    Attributes:
        type: The type name (e.g., number, string, boolean).
    """
    model_config = ConfigDict(extra='forbid')
    type: str = Field(min_length=1)


class FunctionDef(BaseModel):
    """Definition of a callable function.

    Attributes:
        name: The function name.
        description: A description of what the function does.
        parameters: Dictionary mapping parameter names to their type
        definitions.
        returns: The return type definition.
    """
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, TypeDef]
    returns: TypeDef


class FunctionCall(BaseModel):
    """A function call result.

    Attributes:
        prompt: The original natural language request.
        name: The name of the function to call.
        parameters: Dictionary mapping argument names to their values.
    """
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: Dict[str, Any]
