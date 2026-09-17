from llm_sdk import Small_LLM_Model
from tokenizer_utils import load_vocab
from masking import mask_boolean

model = Small_LLM_Model()
vocab = load_vocab(model)

prompt = "Is the sky green? Answer with true or false."
input_ids = model.encode(prompt).tolist()[0]

logits = model.get_logits_from_input_ids(input_ids)
true_id = vocab['true']
false_id = vocab['false']
print("Logit de 'true':", logits[true_id])
print("Logit de 'false':", logits[false_id])

valor, input_ids_actualizado = mask_boolean(input_ids, vocab, model)
print("Valor generado:", valor)
