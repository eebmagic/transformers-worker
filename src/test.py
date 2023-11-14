from worker import encodeWithModel

texts = [
    "this is a short test document",
    "this is a similar test document",
    "def func(params):\n\tprint(f'user passed these params {params}')"
]

modelName = 'Salesforce/codet5p-110m-embedding'

# results = encodeWithModel(texts)
results = encodeWithModel(texts, modelName=modelName)
# print(results)
print(f"Got results: {type(results)}")
print(results.keys())