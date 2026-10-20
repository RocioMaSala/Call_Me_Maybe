"""Command-line entry point: generate function calls for every prompt."""
from llm_sdk import Small_LLM_Model
from .tokenizer_utils import (
    load_vocab,
    build_id_to_token,
    build_unknown_ids,
)
from .masking import (
    mask_number,
    mask_string,
    mask_name,
    mask_boolean,
    write_literal
)
from .prompting import build_prompt_compact, add_json_instruction
from .parser import (
    loading_function_definitions,
    build_name_to_def,
    FunctionDefinition,
    FunctionCallingTest,
    loading_function_calling_test,
    FunctionCallError,
    FunctionCallResult,
)
from .json_utils import escape_json_string
import json
import argparse
import time
import sys
import os
from typing import Any


def generar_json_para_prompt(
    prompt: str,
    definitions: list[FunctionDefinition],
    name_to_def: dict[str, FunctionDefinition],
    vocab: dict[str, int],
    model: Small_LLM_Model,
    unknown_ids: list[int],
) -> dict[str, Any]:
    """Generate the function call for one prompt.

    The JSON text is built in segments. Fixed text (keys, braces,
    separators and the original prompt) is added with write_literal.
    The function name and every argument value are generated with the
    constrained-decoding functions of masking.py, according to the type
    declared for each parameter. The final text is parsed with
    json.loads, and the values of "number" parameters are converted to
    float.

    Args:
        prompt: Request written in natural language.
        definitions: The available functions.
        name_to_def: Maps each function name to its definition.
        vocab: Maps raw token strings to ids.
        model: Model that returns the logits.
        unknown_ids: Positions of the logits that have no text.

    Returns:
        A dictionary with the keys "prompt", "name" and "parameters".

    Raises:
        ValueError: If a parameter has a type that is not "number",
            "string" or "boolean", or if the generated text is not
            valid JSON.
    """
    contexto = add_json_instruction(build_prompt_compact(prompt, definitions))
    input_ids = model.encode(contexto).tolist()[0]
    json_acumulado = ""

    literal_inicio = (
        '{"prompt": "'
        + escape_json_string(prompt)
        + '", "name": "'
    )
    fragmento, input_ids = write_literal(input_ids, model, literal_inicio)
    json_acumulado += fragmento

    nombre_funcion, input_ids = mask_name(
        input_ids, vocab, model, definitions, unknown_ids
        )
    json_acumulado += nombre_funcion

    fragmento, input_ids = write_literal(
        input_ids, model, '", "parameters": {'
        )
    json_acumulado += fragmento

    funcion_elegida = name_to_def[nombre_funcion]
    parametros = funcion_elegida.parameters
    total = len(parametros)

    for i, (nombre_param, esquema) in enumerate(parametros.items()):
        literal_clave = f'"{nombre_param}": '
        if esquema.type == "string":
            literal_clave += '"'
        fragmento, input_ids = write_literal(input_ids, model, literal_clave)
        json_acumulado += fragmento

        if esquema.type == "number" or esquema.type == "integer":
            valor, input_ids = mask_number(input_ids, vocab, model)
        elif esquema.type == "string":
            try:
                valor, input_ids = mask_string(
                    input_ids, vocab, model, unknown_ids
                    )
            except ValueError as e:
                raise ValueError(f"Parámetro {nombre_param!r}: {e}") from e
        elif esquema.type == "boolean":
            valor, input_ids = mask_boolean(input_ids, vocab, model)
        else:
            raise ValueError(f"Tipo de parámetro no soportado: {esquema.type}")
        json_acumulado += valor

        es_ultimo = (i == total - 1)
        cierre = '}' if es_ultimo else ', '
        if esquema.type == "string":
            cierre = '"' + cierre
        fragmento, input_ids = write_literal(input_ids, model, cierre)
        json_acumulado += fragmento

    cierre_final = '}' if total > 0 else '}}'
    fragmento, input_ids = write_literal(input_ids, model, cierre_final)
    json_acumulado += fragmento
    try:
        resultado: dict[str, Any] = json.loads(json_acumulado)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"JSON inválido: {e}. Texto generado: {json_acumulado!r}"
        ) from e
    return resultado


def procesar_todos_los_prompts(
    tests: list[FunctionCallingTest],
    definitions: list[FunctionDefinition],
    name_to_def: dict[str, FunctionDefinition],
    vocab: dict[str, int],
    model: Small_LLM_Model,
    unknown_ids: list[int],
) -> list[FunctionCallResult | FunctionCallError]:
    """Generate a function call for every prompt.

    A prompt that fails does not stop the others: its error is stored
    as a FunctionCallError. A progress line with the time spent is
    printed for each prompt.

    Args:
        tests: The prompts to process.
        definitions: The available functions.
        name_to_def: Maps each function name to its definition.
        vocab: Maps raw token strings to ids.
        model: Model that returns the logits.
        unknown_ids: Positions of the logits that have no text.

    Returns:
        One result per prompt, in the same order as tests.
    """
    resultados: list[FunctionCallResult | FunctionCallError] = []
    for n, test in enumerate(tests, start=1):
        inicio = time.time()
        try:
            datos = generar_json_para_prompt(
                test.prompt, definitions, name_to_def,
                vocab, model, unknown_ids,
            )
            resultado: FunctionCallResult | FunctionCallError = (
                FunctionCallResult(**datos)
            )
            estado = "ok"
        except Exception as e:
            resultado = FunctionCallError(prompt=test.prompt, error=str(e))
            estado = "ERROR"
        resultados.append(resultado)
        print(f"[{n}/{len(tests)}] {estado} en {time.time() - inicio:.1f} s")
    return resultados


def escribir_resultados(
    resultados: list[FunctionCallResult | FunctionCallError],
    ruta_salida: str,
) -> None:
    """Write the successful results to the output file.

    The file is a JSON array. Each object is written by hand so that
    "parameters" stays on one line. Prompts that failed are not written,
    because the output may only have the keys "prompt", "name" and
    "parameters". Their errors are printed on stderr instead. The output
    folder is created if it does not exist.

    Args:
        resultados: The results of every prompt.
        ruta_salida: Path of the output file.
    """
    validos = [r for r in resultados if isinstance(r, FunctionCallResult)]
    bloques = []
    for r in validos:
        prompt = json.dumps(r.prompt, ensure_ascii=False)
        nombre = json.dumps(r.name, ensure_ascii=False)
        params = json.dumps(r.parameters, ensure_ascii=False)
        bloques.append(
            "    {\n"
            f'        "prompt": {prompt},\n'
            f'        "name": {nombre},\n'
            f'        "parameters": {params}\n'
            "    }"
        )
    os.makedirs(os.path.dirname(ruta_salida) or ".", exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(bloques) + "\n]\n")
    if not validos:
        print("AVISO: ningún resultado válido, el fichero queda vacío.",
              file=sys.stderr)
    for fail in resultados:
        if isinstance(fail, FunctionCallError):
            print(f"ERROR en {fail.prompt!r}: {fail.error}", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    """Read the command-line arguments.

    Returns:
        The arguments --functions_definition, --input and --output,
        each with a default path in data/input or data/output.
    """
    parser = argparse.ArgumentParser(
        description="Function calling con constrained decoding"
        )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
    )
    return parser.parse_args()


def main() -> None:
    """Run the whole program.

    Loads the model and the vocabulary, reads the input files,
    generates a function call for every prompt and writes the result.
    """
    args = parse_args()

    definitions = loading_function_definitions(args.functions_definition)
    tests = loading_function_calling_test(args.input)
    if not tests:
        print(f"Error: {args.input} no contiene ningún prompt.",
              file=sys.stderr)
        sys.exit(1)

    name_to_def = build_name_to_def(definitions)

    model = Small_LLM_Model()
    vocab = load_vocab(model)
    id_to_token = build_id_to_token(vocab)

    probe_ids = model.encode("Test").tolist()[0]
    logits_size = len(model.get_logits_from_input_ids(probe_ids))
    unknown_ids = build_unknown_ids(id_to_token, logits_size)

    definitions = loading_function_definitions(args.functions_definition)
    tests = loading_function_calling_test(args.input)

    name_to_def = build_name_to_def(definitions)

    resultados = procesar_todos_los_prompts(
        tests, definitions, name_to_def, vocab, model, unknown_ids
    )
    escribir_resultados(resultados, args.output)


if __name__ == "__main__":
    main()
