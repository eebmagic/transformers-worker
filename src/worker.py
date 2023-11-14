from transformers import AutoModel, AutoTokenizer
import torch
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"RUNNING ON DEVICE: {device}")

def codeBertFunc(model, tokenized):
    with torch.no_grad():
        outputs = model(**tokenized).last_hidden_state
        return outputs.mean(dim=1).cpu().numpy()


def t5Func(model, tokenized):
    tokenized = tokenized.to(device)
    results = model(**tokenized)
    cpu = results.cpu().detach().numpy()

    return cpu


modelMap = {
    'microsoft/codebert-base': (
        '/usr/src/app/models/microsoft/codebert-base-model',
        '/usr/src/app/models/microsoft/codebert-base-tokenizer',
        codeBertFunc
    ),
    'Salesforce/codet5p-110m-embedding': (
        '/usr/src/app/models/Salesforce/codet5p-110m-embedding-model',
        '/usr/src/app/models/Salesforce/codet5p-110m-embedding-tokenizer',
        t5Func
    )
}


def encodeWithModel(texts, modelName='microsoft/codebert-base'):
    if modelName in modelMap:
        print(f"LOADING MODEL FROM LOCAL FILE")
        modelDir, tokenizerDir, func = modelMap[modelName]
        model = AutoModel.from_pretrained(modelDir, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(tokenizerDir, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(modelName, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(modelName, trust_remote_code=True)
        func = codeBertFunc

    # Tokenize
    print("TOKENIZING...")
    tokenizedInput = tokenizer.batch_encode_plus(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)

    # Put through the model
    print(f"ENCODING WITH MODEL...")
    start = time.time()
    embeddings = func(model, tokenizedInput)
    duration = time.time() - start

    # Build and return output
    print(f"RETURNING OUTPUT...")
    output = {
        "embeddings": embeddings.tolist(),
        "duration": duration,
        "modelName": modelName,
    }
    return output