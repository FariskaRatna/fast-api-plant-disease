import numpy as np
import onnxruntime
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from io import BytesIO
from scipy.special import softmax
from typing import Any, Dict, Tuple


class ONNXPredictor:
    def __init__(self, model_path: str, class_mapping: Dict[int, str]) -> None:
        """
        Constructor method to initialize the ONNXPredictor object.

        Parameters:
        - model_path (str): Path to the ONNX model file
        - class_mapping (Dict[int, str]): Mapping of class indices to class names

        Initializes an InferenceSession with the ONNX model, extracts input and output information,
        defines data transformation pipeline, and sets class mapping.
        """

        self.session = onnxruntime.InferenceSession(model_path)
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape
        output_info = self.session.get_outputs()[0]
        self.output_name = output_info.name
        self.output_shape = output_info.shape
        self.transform = Compose([Resize((224, 224)), ToTensor()])
        self.class_mapping = class_mapping

    def preprocessing(self, image: bytes) -> np.ndarray:
        """
        Preprocessing method to transform input image data.

        Parameters:
        - image (bytes): Input image data in bytes format

        Returns:
        - np.ndarray: Preprocessed image data as a numpy array
        """

        image = Image.open(BytesIO(image))
        image = self.transform(image)
        return np.expand_dims(image.numpy(), axis=0)

    def prediction(self, image: bytes) -> Tuple[str, float, int]:
        """
        Prediction method to perform inference on input image.

        Parameters:
        - image (bytes): Input image data in bytes format

        Returns:
        - Tuple[str, float, int]: Predicted class name, probability, and index
        """
        
        input_data = self.preprocessing(image)
        output = self.session.run([self.output_name], {self.input_name: input_data})
        probabilities = softmax(output[0], axis=1)
        predicted_class_index = int(np.argmax(probabilities))
        predicted_class_prob = float(probabilities[0][predicted_class_index])
        predicted_class_name = self.class_mapping.get(predicted_class_index, "Unknowwn")
        return predicted_class_name, predicted_class_prob, predicted_class_index
