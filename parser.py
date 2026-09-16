from pydantic import BaseModel, Field, ValidationError
import json
import sys


class ParameterSchema(BaseModel):
    type: str

class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema

class FunctionCallingTest(BaseModel):
    prompt: str

tests: list[FunctionCallingTest]
definitions: list[FunctionDefinition]

def loading_function_definitions(path: str) -> list[FunctionDefinition]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            definitions = []
            for item in raw_data:
                parsed_item = FunctionDefinition(**item)
                definitions.append(parsed_item)          
            return definitions
    except FileNotFoundError:
        print(f"Error: no se encontró el fichero {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: el fichero {path} no contiene JSON válido: {e}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"Error: el fichero {path} no cumple el formato esperado: {e}", file=sys.stderr)
        sys.exit(1)

def loading_function_calling_test(path: str) -> list[FunctionCallingTest]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
            tests = []
            for item in raw_data:
                parsed_item = FunctionCallingTest(**item)
                tests.append(parsed_item)          
            return tests
    except FileNotFoundError:
        print(f"Error: no se encontró el fichero {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: el fichero {path} no contiene JSON válido: {e}", file=sys.stderr)
        sys.exit(1)
    except ValidationError as e:
        print(f"Error: el fichero {path} no cumple el formato esperado: {e}", file=sys.stderr)
        sys.exit(1)