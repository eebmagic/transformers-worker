import runpod

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

from worker import encodeWithModel

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']

    texts = job_input.get('texts')
    modelName = job_input.get('modelName')


    output = encodeWithModel(texts, modelName)

    return output


print(f"Starting runpod")
runpod.serverless.start({"handler": handler})
