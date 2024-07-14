from scripts.classifier import Classifier
import torch
import onnx

"""
Main function for exporting a PyTorch model to ONNX format.

Instantiates a Classifier model with 8 output classes, loads pretrained weights,
sets the model to evaluation mode, creates a dummy input tensor, and exports
the model to ONNX format.
"""

def main():
    model = Classifier(n_classes=8)
    model.load_state_dict(torch.load("./model/Modified_Efficient.pt"))
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            "./Efficient.onnx",
            input_names=["input"],
            output_names=["output"],
            verbose=True,
        )


if __name__ == "__main__":
    main()
