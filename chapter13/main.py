"""Fantasy acquisition API"""

from fastapi import FastAPI
import onnxruntime as rt
import numpy as np
from schemas import FantasyAcquisitionFeatures, PredictionOutput

api_description = """
This API predicts the range of costs to acquire a player in fantasy football

The endpoints are grouped into the following categories:

## Analytics
Get information about health of the API.

## Prediction
Get predictions of player acquisition cost.
"""

# Load the ONNX model
sess_10 = rt.InferenceSession("acquisition_model_10.onnx", 
                              providers=["CPUExecutionProvider"])
sess_50 = rt.InferenceSession("acquisition_model_50.onnx", 
                              providers=["CPUExecutionProvider"])
sess_90 = rt.InferenceSession("acquisition_model_90.onnx", 
                              providers=["CPUExecutionProvider"])

# Get the input and output names of the model
input_name_10 = sess_10.get_inputs()[0].name
label_name_10 = sess_10.get_outputs()[0].name
input_name_50 = sess_50.get_inputs()[0].name
label_name_50 = sess_50.get_outputs()[0].name
input_name_90 = sess_90.get_inputs()[0].name
label_name_90 = sess_90.get_outputs()[0].name

# FastAPI constructor with additional details added for OpenAPI Specification
app = FastAPI(
    description=api_description,
    title="Fantasy acquisition API",
    version="0.1",
)

@app.get(
    "/",
    summary="Check to see if the Fantasy acquisition API is running",
    description="""Use this endpoint to check if the API is running. You can also check it first before making other calls to be sure it's running.""",
    response_description="A JSON record with a message in it. If the API is running the message will say successful.",
    operation_id="v0_health_check",
    tags=["analytics"],
)
def root():
    return {"message": "API health check successful"}