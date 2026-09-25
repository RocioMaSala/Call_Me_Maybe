from llm_sdk import Small_LLM_Model
from tokenizer_utils import load_vocab
from masking import mask_string

model = Small_LLM_Model()
vocab = load_vocab(model)

context = (
    'User request: Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS\n'
    '{"name": "fn_replace_numbers", "parameters": {"text": "'
)

input_ids = model.encode(context).tolist()[0]

valor, input_ids_actualizado = mask_string(input_ids, vocab, model)
print("String generado:", repr(valor))