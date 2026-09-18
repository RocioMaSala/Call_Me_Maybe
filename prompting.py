from parser import FunctionDefinition, ParameterSchema

def format_parameters(parameters: dict[str, ParameterSchema]) -> str:
    prompt_parameters = [f"{name}: {value.type}" for name, value in parameters.items]
    return ", ".join(prompt_parameters)

def format_function_compact(func: FunctionDefinition) -> str:
    result = f"- {func.name}({format_parameters(func.parameters)}): {func.description}"
    return result

def build_functions_section(definitions: list[FunctionDefinition]) -> str:
    prompt_functions = f"Available functions:\n"
    for function in definitions:
        prompt_functions += f"{format_function_compact(function)}\n"
    return prompt_functions

def build_prompt_compact(user_prompt: str, definitions: list[FunctionDefinition]) -> str:
    prompt_complete_compact = f"{build_functions_section(definitions)}"
    prompt_complete_compact += f"\nUser request: {user_prompt}"
    return prompt_complete_compact

# Analizar cuando tenga la máscara, qué tipo de prompt es mejor que le pase a la máquina.
def format_function_explicit(func: FunctionDefinition) -> str:
    result = f"Function name: {func.name} \nParameters: {format_parameters(func.parameters)} \nDescription: {func.description}"
    return result

def build_functions_section_explicit(definitions: list[FunctionDefinition]) -> str:
    prompt_functions = f"Available functions:\n"
    for function in definitions:
        prompt_functions += f"{format_function_explicit(function)}\n"
    return prompt_functions

def build_prompt_explicit(user_prompt: str, definitions: list[FunctionDefinition]) -> str:
    prompt_complete_explicit = f"{build_functions_section_explicit(definitions)}"
    prompt_complete_explicit += f"\nUser request: {user_prompt}"
    return prompt_complete_explicit

