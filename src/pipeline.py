import torch
from src.models.triplet_model import TripletAttentionModel


def forward_pipeline(input_tensor):
    model = TripletAttentionModel()
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)

    return output


if __name__ == "__main__":
    dummy_input = torch.randn(1, 3, 224, 224)
    output = forward_pipeline(dummy_input)

    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)
