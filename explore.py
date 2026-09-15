from llm_sdk import Small_LLM_Model
import json

model = Small_LLM_Model()
vocab_path = model.get_path_to_vocab_file()
print(vocab_path)

with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

print(type(vocab))
print(len(vocab))

# Busquemos tokens que sean puramente numéricos (de cualquier longitud)
numeric_tokens = {t: i for t, i in vocab.items() if t.isdigit()}
print(f"Tokens puramente numéricos: {len(numeric_tokens)}")
print(list(numeric_tokens.items())[:15])

# Busquemos cómo se representan caracteres clave de JSON
for target in ['{', '}', '"', ':', ',', '[', ']']:
    matches = {t: i for t, i in vocab.items() if t == target}
    print(target, "->", matches)

for target in ['true', 'false', 'True', 'False']:
    matches = {t: i for t, i in vocab.items() if t == target}
    print(repr(target), "->", matches)