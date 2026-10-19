"""Build the text context that is sent to the model."""
from .parser import FunctionDefinition, ParameterSchema


def add_json_instruction(prompt: str) -> str:
    """Append a formatting instruction and one worked example.

    The example shows a complete function call for a request that is
    not in the test files. It is written after the text received, so
    the context ends with the example and not with the user request.

    Args:
        prompt: Context built so far.

    Returns:
        The same context followed by the instruction and the example.
    """
    instruccion = (
        "\n\nRespond with a single JSON object, "
        "with no text before or after it.\n"
        "Example:\n"
        "User request: Replace all spaces in 'a b c' with '-'\n"
        'Response: {"prompt": "Replace all spaces in \'a b c\' with \'-\'", '
        '"name": "fn_substitute_string_with_regex", "parameters": '
        '{"source_string": "a b c", "regex": "\\\\s", "replacement": "-"}}\n'
    )
    return prompt + instruccion


def format_parameters(parameters: dict[str, ParameterSchema]) -> str:
    """Write the parameters of a function as a signature.

    Args:
        parameters: Maps each parameter name to its type.

    Returns:
        The parameters as "name: type", separated by commas.
    """
    prompt_parameters = [
        f"{name}: {value.type}" for name, value in parameters.items()
        ]
    return ", ".join(prompt_parameters)


def format_function_compact(func: FunctionDefinition) -> str:
    result = (
        f"- {func.name}({format_parameters(func.parameters)}): "
        f"{func.description}"
        )
    """Write one function as a single line.

    The return type is not included, because it does not help to choose
    a function or its arguments.

    Args:
        func: Function to describe.

    Returns:
        A line like "- name(a: number, b: number): description".
    """
    return result


def build_functions_section(definitions: list[FunctionDefinition]) -> str:
    """List every available function, one per line.

    No function is filtered out: the model has to choose among all of
    them.

    Args:
        definitions: The available functions.

    Returns:
        A header "Available functions:" followed by one line per
        function.
    """
    prompt_functions = "Available functions:\n"
    for function in definitions:
        prompt_functions += f"{format_function_compact(function)}\n"
    return prompt_functions


def build_prompt_compact(
        user_prompt: str, definitions: list[FunctionDefinition]
        ) -> str:
    """Build the context for one request.

    Args:
        user_prompt: Request written in natural language.
        definitions: The available functions.

    Returns:
        The list of functions followed by "User request: <prompt>".
    """
    prompt_complete_compact = f"{build_functions_section(definitions)}"
    prompt_complete_compact += f"\nUser request: {user_prompt}"
    return prompt_complete_compact
