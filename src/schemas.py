"""
Defines the project's data structures and validation models.

Responsibilities:

Represent function definitions.
Represent function parameters.
Represent generated function calls.
Provide validation and serialization helpers.
Keep data handling consistent across the project.

"""
from pydantic import BaseModel, field_validator


class PromptEntry(BaseModel):
    """
    To validate function_calling_tests.json
    or any file used as input prompt
    """
    prompt: str


class ParameterSchema(BaseModel):
    """
    Represents the type for each parameter on the 
    functions_definition.json - JSON file:
    "string", "number", "integer", and "boolean"
    """
    type: str

    @field_validator('type')
    @classmethod
    def validate_parameter_schema(cls, value: str) -> str:
        types: list[str] = ["string", "number", "integer", "boolean"]
        if value not in types:
            raise ValueError("Invalid type on ParameterSchema.")
        return value


class FunctionDefinition(BaseModel):
    """
    Represents each field on JSON:
    functions_definition.json
    name,description,parameters,returns
    """
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema


class FunctionCall(BaseModel):
    """
    Represents the final output,
    To be written on the JSON file
    function_calling_results.json
    """
    prompt: str
    name: str
    parameters: dict[str, float | str | bool]
