import sys
sys.path.insert(0, "llm_sdk")

from llm_sdk import Small_LLM_Model
import json

model = Small_LLM_Model()

# 1. Ver o vocabulário
vocab_path = model.get_path_to_vocab_file()
with open(vocab_path) as f:
    vocab = json.load(f)

# vocab é {token_string: token_id} — inverter para usar por ID
id_to_token = {v: k for k, v in vocab.items()}

print(f"Vocabulary size: {len(vocab)}")

# 2. Encontrar tokens importantes
print("\nTokens estruturais:")
for token, id in vocab.items():
    if token in ["{", "}", '"', ":", ","]:
        print(f"  {repr(token)} → ID {id}")

# 3. Ver como tokeniza nomes de funções
print("\nTokenização de fn_add_numbers:")
ids = model.encode("fn_add_numbers")[0].tolist()
for i in ids:
    print(f"  ID {i} → {repr(id_to_token[i])}")

print("\nTokenização de fn_greet:")
ids = model.encode("fn_greet")[0].tolist()
for i in ids:
    print(f"  ID {i} → {repr(id_to_token[i])}")