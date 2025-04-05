from typing import List

import torch

from model import RegressionModel
from dirs import checkpoint_path

@torch.no_grad()
def predict(input_features: List[float]):
    # load the checkpoint from the correct path
    checkpoint = torch.load(checkpoint_path)

    input_size = checkpoint.get("input_size")
    hidden_size = checkpoint.get("hidden_size")
    x_mean = checkpoint.get("x_mean")
    x_std = checkpoint.get("x_std")
    y_mean = checkpoint.get("y_mean")
    y_std = checkpoint.get("y_std")
    model_state_dict = checkpoint.get("model_state_dict")

    # Instantiate the model and load the state dict
    model = RegressionModel(input_size, hidden_size)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Input features is a list of floats. We have to convert it to tensor of the correct shape
    x = torch.tensor(input_features).unsqueeze(0)

    # Now we have to do the same normalization we did when training:
    x = (x - x_mean) / x_std

    # We get the output of the model and we print it
    output = model(x)

    # We have to revert the target normalization that we did when training:
    output = (output * y_std) + y_mean
    print(f"The predicted price is: ${output.item()*1000:.2f}")
