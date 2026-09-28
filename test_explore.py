from llm_sdk import Small_LLM_Model
from tokenizer_utils import load_vocab, build_id_to_token, build_unknown_ids
from masking import mask_number, mask_string, mask_name
from prompting import build_prompt_compact
from parser import loading_function_definitions

model = Small_LLM_Model()
vocab = load_vocab(model)
id_to_token = build_id_to_token(vocab)
definitions = loading_function_definitions("data/input/functions_definition.json")

probe_ids = model.encode("Test").tolist()[0]
logits_size = len(model.get_logits_from_input_ids(probe_ids))
unknown_ids = build_unknown_ids(id_to_token, logits_size)

# 1. mask_number
ctx = build_prompt_compact("What is the sum of 265 and 345?", definitions)
ctx += '\n{"name": "fn_add_numbers", "parameters": {"a": '
ids = model.encode(ctx).tolist()[0]
valor, _ = mask_number(ids, vocab, model, unknown_ids)
print("Número:", repr(valor))

# 2. mask_string
ctx = 'User request: Reverse the string \'hello\'\n{"name": "fn_reverse_string", "parameters": {"s": "'
ids = model.encode(ctx).tolist()[0]
valor, _ = mask_string(ids, vocab, model, unknown_ids)
print("String:", repr(valor))

# 3. mask_name
ctx = build_prompt_compact("What is the sum of 2 and 3?", definitions)
ctx += '\n{"name": "'
ids = model.encode(ctx).tolist()[0]
valor, _ = mask_name(ids, vocab, model, definitions, unknown_ids)
print("Nombre:", repr(valor))