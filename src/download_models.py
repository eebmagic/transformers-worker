from transformers import AutoModel, AutoTokenizer
import sys

def downloadModel(modelName):
    model = AutoModel.from_pretrained(modelName, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True)

    modelFile = f"{modelName}-model"
    tokenizerFile = f"{modelName}-tokenizer"

    modelExportPath = f"../model-downloads/{modelFile}"
    tokenizerExportPath = f"../model-downloads/{tokenizerFile}"

    model.save_pretrained(modelExportPath)
    tokenizer.save_pretrained(tokenizerExportPath)

    modelPath = f'/usr/src/app/models/{modelFile}'
    tokenizerPath = f'/usr/src/app/models/{tokenizerFile}'

    return modelPath, tokenizerPath

if len(sys.argv) > 1:
    modelname = sys.argv[1]
else:
    modelname = input(f"Type the model name to download: ").strip()
modelPath, tokenizerPath = downloadModel(modelname)

print(f"Finsihed downloading. Add this to the worker code:")
print(f"'{modelname}': (")
print(f"    '{modelPath},")
print(f"    '{tokenizerPath},")
print(f"    codeBertFunc")
print(f"),")