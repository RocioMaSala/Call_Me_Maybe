from llm_sdk import Small_LLM_Model
from tokenizer_utils import load_vocab, build_id_to_token, build_unknown_ids
from parser import loading_function_definitions, build_name_to_def
from call_me_maybe import generar_json_para_prompt

model = Small_LLM_Model()
vocab = load_vocab(model)
id_to_token = build_id_to_token(vocab)
definitions = loading_function_definitions("data/input/functions_definition.json")
name_to_def = build_name_to_def(definitions)

probe_ids = model.encode("Test").tolist()[0]
logits_size = len(model.get_logits_from_input_ids(probe_ids))
unknown_ids = build_unknown_ids(id_to_token, logits_size)

resultado = generar_json_para_prompt(
    "What is the sum of 2 and 3?",
    definitions,
    name_to_def,
    vocab,
    model,
    unknown_ids,
)
print(resultado)