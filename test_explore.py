from llm_sdk import Small_LLM_Model
from src.tokenizer_utils import load_vocab, build_id_to_token

m = Small_LLM_Model()
t = build_id_to_token(load_vocab(m))
texto = '"regex": "\\\\s"'  # texto real: "regex": "\\s"
print([t[i] for i in m.encode(texto).tolist()[0]])