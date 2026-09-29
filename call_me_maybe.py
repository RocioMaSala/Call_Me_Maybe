from llm_sdk import Small_LLM_Model
from tokenizer_utils import load_vocab, build_id_to_token, build_unknown_ids
from masking import mask_number, mask_string, mask_name, mask_boolean, mask_literal
from prompting import build_prompt_compact
from parser import loading_function_definitions, build_name_to_def, FunctionDefinition, FunctionCallingTest
from json_utils import escape_json_string
import json
import argparse


def generar_json_para_prompt (
        prompt: str,
        definitions: list[FunctionDefinition],
        name_to_def: dict[str, FunctionDefinition],
        vocab: dict[str, int],
        model: Small_LLM_Model,
        unknown_ids: list[int]
        ) -> dict[str, object]:
    contexto = build_prompt_compact(prompt, definitions)
    input_ids = model.encode(contexto).tolist()[0]
    json_acumulado = ""

    literal_inicio = '{\n   "prompt": "' + escape_json_string(prompt) + '",\n   "name": "'
    fragmento, input_ids = mask_literal(input_ids, vocab, model, literal_inicio, unknown_ids)
    json_acumulado += fragmento

    nombre_funcion, input_ids = mask_name(input_ids, vocab, model, definitions, unknown_ids)
    json_acumulado += nombre_funcion

    fragmento, input_ids = mask_literal(input_ids, vocab, model, '",\n   "parameters": {', unknown_ids)
    json_acumulado += fragmento

    funcion_elegida = name_to_def[nombre_funcion]
    parametros = funcion_elegida.parameters
    total = len(parametros)

    for i, (nombre_param, esquema) in enumerate(parametros.items()):
        literal_clave = f'"{nombre_param}": '
        fragmento, input_ids = mask_literal(input_ids, vocab, model, literal_clave, unknown_ids)
        json_acumulado += fragmento

        if esquema.type == "number":
            valor, input_ids = mask_number(input_ids, vocab, model, unknown_ids)
        elif esquema.type == "string":
            valor, input_ids = mask_string(input_ids, vocab, model, unknown_ids)
        elif esquema.type == "boolean":
            valor, input_ids = mask_boolean(input_ids, vocab, model)
        else:
            raise ValueError(f"Tipo de parámetro no soportado: {esquema.type}")
        json_acumulado += valor

        es_ultimo = (i == total - 1)
        literal_cierre = '}' if es_ultimo else ', '
        fragmento, input_ids = mask_literal(input_ids, vocab, model, literal_cierre, unknown_ids)
        json_acumulado += fragmento
        
    fragmento, input_ids = mask_literal(input_ids, vocab, model, '\n}', unknown_ids)
    json_acumulado += fragmento

    resultado = json.loads(json_acumulado)
    return resultado

def procesar_todos_los_prompts(
        tests: list[FunctionCallingTest],
        definitions: list[FunctionDefinition],
        name_to_def: dict[str, FunctionDefinition],
        vocab: dict[str, int],
        model: Small_LLM_Model, 
        unknown_ids: list[int],
) -> list[dict]: 
    resultados = []
    for test in tests:
        try:
            resultado = generar_json_para_prompt(test.prompt, definitions, name_to_def, vocab, model, unknown_ids)
            resultados.append(resultado)
        except Exception as e:
            resultado_error = {"prompt": test.prompt, "error": str(e)}
            resultados.append(resultado_error)
    return resultados

def escribir_resultados(resultados: list[dict], ruta_salida: str) -> None:
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultados, f)

import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Function calling con constrained decoding")
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Ruta al fichero de definición de funciones",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Ruta al fichero de prompts de entrada",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Ruta al fichero de salida",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = Small_LLM_Model()
    vocab = load_vocab(model)
    id_to_token = build_id_to_token(vocab)

    # HUECO 1: calcular unknown_ids (necesitas logits_size primero)
    ???

    # HUECO 2: cargar definitions y tests, usando las rutas de args
    ???

    # HUECO 3: construir name_to_def
    ???

    # HUECO 4: procesar todos los prompts
    ???

    # HUECO 5: escribir el resultado
    ???



if __name__ == "__main__":
    main()