import time
from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import (
    load_vocab,
    build_id_to_token,
    build_id_to_text,
    build_unknown_ids,
)
from src.parser import loading_function_definitions, build_name_to_def
from src.prompting import build_prompt_compact, add_json_instruction
from src.call_me_maybe import generar_json_para_prompt

model = Small_LLM_Model()
vocab = load_vocab(model)
id_to_token = build_id_to_token(vocab)
id_to_text = build_id_to_text(id_to_token)
definitions = loading_function_definitions("data/input/functions_definition.json")
name_to_def = build_name_to_def(definitions)

probe_ids = model.encode("Test").tolist()[0]
logits_size = len(model.get_logits_from_input_ids(probe_ids))
unknown_ids = build_unknown_ids(id_to_token, logits_size)

prompt = "Reverse the string 'world'"

original = model.get_logits_from_input_ids
stats = {"calls": 0, "seconds": 0.0}


def medido(ids):
    t = time.time()
    salida = original(ids)
    dt = time.time() - t
    stats["calls"] += 1
    stats["seconds"] += dt
    print(f"llamada {stats['calls']}: {dt:.1f} s, {len(ids)} tokens", flush=True)
    return salida


model.get_logits_from_input_ids = medido

inicio = time.time()
resultado = generar_json_para_prompt(
    prompt, definitions, name_to_def, vocab, model, unknown_ids, id_to_text
)
total = time.time() - inicio

contexto = add_json_instruction(build_prompt_compact(prompt, definitions))
n_tokens = len(model.encode(contexto).tolist()[0])

print(resultado)
print(f"Llamadas al modelo: {stats['calls']}")
print(f"Tiempo dentro del modelo: {stats['seconds']:.1f} s")
print(f"Tiempo total: {total:.1f} s")
print(f"Tokens del contexto inicial: {n_tokens}")