import uvicorn
from fastapi import FastAPI, File, UploadFile
from prediction_onnx import ONNXPredictor

app = FastAPI()

class_mapping = {
    0: "cucumber_health",
    1: "cucumber_unhealthy",
    2: "potato_early_blight",
    3: "potato_healthy",
    4: "potato_late_blight",
    5: "tomato_early_blight",
    6: "tomato_healthy",
    7: "tomato_late_blight",
}

onnx_model = "./onnx/model/EfficientNet.onnx"
plant_classifier = ONNXPredictor(onnx_model, class_mapping)


@app.post("/prediction/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint for making predictions on uploaded images.

    Accepts image files as input, performs predictions using the ONNXPredictor,
    and returns the predicted class index, name, and probability.
    """
    
    contents = await file.read()

    predicted_class_name, predicted_class_proba, predicted_class_index = (
        plant_classifier.prediction(contents)
    )
    return {
        "Predicted Class Index": predicted_class_index,
        "Predicted Class Name ": predicted_class_name,
        "Predicted Class Probability ": predicted_class_proba,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
