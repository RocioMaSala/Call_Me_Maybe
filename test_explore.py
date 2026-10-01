from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import load_vocab, build_id_to_token
from src.prompting import build_prompt_compact, add_json_instruction
from src.parser import loading_function_definitions

model = Small_LLM_Model()
vocab = load_vocab(model)
id_to_token = build_id_to_token(vocab)
definitions = loading_function_definitions("data/input/functions_definition.json")

contexto = add_json_instruction(build_prompt_compact("What is the sum of 2 and 3?", definitions))
input_ids = model.encode(contexto).tolist()[0]

logits = model.get_logits_from_input_ids(input_ids)
indices_ordenados = sorted(range(len(logits)), key=lambda i: logits[i], reverse=True)
top_8 = indices_ordenados[:8]

for id_token in top_8:
    texto = id_to_token.get(id_token, "<<TOKEN DESCONOCIDO/ESPECIAL>>")
    print(f"ID: {id_token}, texto: {texto!r}, logit: {logits[id_token]:.3f}")