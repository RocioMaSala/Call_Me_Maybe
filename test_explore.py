import time
from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import (
    load_vocab,
    build_id_to_token,
    build_id_to_text,
    build_unknown_ids,
)
from src.parser import loading_function_definitions, build_name_to_def
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

prompts = [
    "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "Greet shrek",
    "Reverse the string 'world'",
    "What is the sum of 265 and 345?",
    "Calculate the square root of 144",
]

original = model.get_logits_from_input_ids
llamadas = [0]


def medido(ids):
    llamadas[0] += 1
    return original(ids)


model.get_logits_from_input_ids = medido

for prompt in prompts:
    llamadas[0] = 0
    inicio = time.time()
    try:
        r = generar_json_para_prompt(
            prompt, definitions, name_to_def, vocab, model, unknown_ids, id_to_text
        )
        r = {k: v for k, v in r.items() if k != "prompt"}
    except Exception as e:
        r = f"ERROR: {e}"
    print(f"\n{prompt}")
    print(f"  -> {r}")
    print(f"  {llamadas[0]} llamadas, {time.time() - inicio:.1f} s", flush=True)