from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import load_vocab, build_id_to_token
from src.prompting import build_prompt_compact, add_json_instruction

model = Small_LLM_Model()
vocab = load_vocab(model)
id_to_token = build_id_to_token(vocab)

from src.parser import loading_function_definitions
definitions = loading_function_definitions("data/input/functions_definition.json")

contexto = add_json_instruction(build_prompt_compact("What is the sum of 2 and 3?", definitions))
contexto += '{\n '  # el progreso que ya sabemos que el modelo alcanzó
input_ids = model.encode(contexto).tolist()[0]

logits = model.get_logits_from_input_ids(input_ids)
indices_ordenados = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)

espacio_id = vocab[' '] if ' ' in vocab else None
print("ID del espacio puro:", espacio_id)

# Busca en qué posición del ranking está el espacio (o el espacio con marcador Ġ)
for posicion, id_token in enumerate(indices_ordenados[:200]):
    texto = id_to_token.get(id_token, "")
    if texto.strip('Ġ') == '' and len(texto) > 0:  # es un token que, limpio, es solo espacio(s)
        print(f"Posición {posicion}: ID {id_token}, texto {texto!r}, logit {logits[id_token]:.3f}")