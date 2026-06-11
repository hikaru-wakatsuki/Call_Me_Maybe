from pydantic import BaseModel, ConfigDict, Field
from typing import Dict


class Prompt(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1)


class TypeDef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: str = Field(min_length=1)


class FunctionDef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, TypeDef]
    returns: TypeDef


class FunctionCall(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1)
    name: str = Field(min_length=1)
    parameters: Dict[str, float | str | bool]
