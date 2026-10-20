"""Pydantic models and loaders for the input and output data."""
from pydantic import BaseModel, ValidationError, ConfigDict
from typing import Any
import json
import sys


class ParameterSchema(BaseModel):
    """Type declaration of a parameter or of a return value.

    Attributes:
        type: Name of the type, for example "number" or "string".
    """
    type: str


class FunctionDefinition(BaseModel):
    """A function that the model can choose to call.

    Attributes:
        name: Name of the function.
        description: What the function does.
        parameters: Maps each parameter name to its type.
        returns: Type of the value the function returns.
    """
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema


class FunctionCallingTest(BaseModel):
    """One natural-language prompt read from the input file.

    Attributes:
        prompt: The request written in natural language.
    """
    model_config = ConfigDict(extra="forbid")
    prompt: str


class FunctionCallResult(BaseModel):
    """A function call generated for one prompt.

    Attributes:
        prompt: The original request.
        name: Name of the chosen function.
        parameters: Maps each parameter name to its generated value.
    """
    prompt: str
    name: str
    parameters: dict[str, Any]


class FunctionCallError(BaseModel):
    """A prompt for which no function call could be generated.

    Attributes:
        prompt: The original request.
        error: Message that explains why the generation failed.
    """
    prompt: str
    error: str


def loading_function_definitions(path: str) -> list[FunctionDefinition]:
    """Read and validate the file with the function definitions.

    If the file cannot be read or is not valid, an error message is
    printed on stderr and the program ends.

    Args:
        path: Path to the JSON file with a list of definitions.

    Returns:
        The definitions, in the order of the file.

    Raises:
        SystemExit: If the file cannot be opened, is not valid JSON, is
        not a non-empty list of objects, or an item does not match
        FunctionDefinition.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            print(
                f"Error: el fichero {path} debe contener una lista "
                f"de objetos",
                file=sys.stderr
                )
            sys.exit(1)
        definitions = []
        for item in raw_data:
            if not isinstance(item, dict):
                print(
                    f"Error: el fichero {path} tiene un elemento que "
                    f"no es un objeto: {item!r}",
                    file=sys.stderr
                    )
                sys.exit(1)
            definitions.append(FunctionDefinition(**item))
        if not definitions:
            print(f"Error: el fichero {path} no contiene ninguna "
                  f"función.", file=sys.stderr)
            sys.exit(1)
        return definitions
    except (OSError, UnicodeDecodeError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(
            f"Error: el fichero {path} no contiene "
            f"JSON válido: {e}", file=sys.stderr
            )
        sys.exit(1)
    except ValidationError as e:
        print(
            f"Error: el fichero {path} no cumple el "
            f"formato esperado: {e}", file=sys.stderr
            )
        sys.exit(1)


def build_name_to_def(
        definitions: list[FunctionDefinition]
        ) -> dict[str, FunctionDefinition]:
    """Index the definitions by function name.

    Args:
        definitions: The function definitions.

    Returns:
        A dictionary that maps each function name to its definition.
    """
    return {func.name: func for func in definitions}


def loading_function_calling_test(path: str) -> list[FunctionCallingTest]:
    """Read and validate the file with the prompts to process.

    If the file cannot be read or is not valid, an error message is
    printed on stderr and the program ends.

    Args:
        path: Path to the JSON file with a list of prompts.

    Returns:
        The prompts, in the order of the file.

    Raises:
        SystemExit: If the file cannot be opened, is not valid JSON, or
            an item does not match FunctionCallingTest.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            print(
                f"Error: el fichero {path} debe contener una lista "
                f"de objetos",
                file=sys.stderr
                )
            sys.exit(1)
        tests = []
        for item in raw_data:
            if not isinstance(item, dict):
                print(
                    f"Error: el fichero {path} tiene un elemento que "
                    f"no es un objeto: {item!r}",
                    file=sys.stderr
                    )
                sys.exit(1)
            tests.append(FunctionCallingTest(**item))
        return tests
    except OSError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(
            f"Error: el fichero {path} no contiene JSON válido: {e}",
            file=sys.stderr
            )
        sys.exit(1)
    except ValidationError as e:
        print(
            f"Error: el fichero {path} no cumple el formato esperado: {e}",
            file=sys.stderr
            )
        sys.exit(1)
