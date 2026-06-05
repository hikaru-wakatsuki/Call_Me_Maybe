from pydantic import BaseModel, ConfigDict
from enum import Enum
from typing import Dict


class Prompt(BaseModel):
    prompt: str


class Type(str, Enum):
    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"


class TypeDef(BaseModel):
    model_config = ConfigDict(extra='forbid')
    type: Type


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, TypeDef]
    returns: TypeDef


class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, float | str | bool]
