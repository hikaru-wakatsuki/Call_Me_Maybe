from llm_sdk import Small_LLM_Model

model = Small_LLM_Model()

for s in ["{", "}", ",", ": ", '"a"']:
    ids = model.encode(s).tolist()[0]
    print(f"'{s}': {ids}")
