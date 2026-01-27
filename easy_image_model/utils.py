import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


def preprocess_image(img_path, img_size=224):
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    if img_size <= 0:
        raise ValueError("img_size must be positive")

    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Could not open image {img_path}: {e}")

    _image_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = _image_transform(img).unsqueeze(0)  # [1, C, H, W]
    tensor = tensor.view(1, -1)  # flatten to [1, C*H*W] = [1, 150528]
    return tensor


class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, num_categories):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for nodes in hidden_layers:
            layers.append(nn.Linear(prev_dim, nodes))
            layers.append(nn.ReLU())  # activation after each hidden layer
            prev_dim = nodes

        layers.append(nn.Linear(prev_dim, num_categories))  # final output layer
        # No activation here — use BCEWithLogitsLoss for multi-label
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def model_to_dict(model):
    model_dict = {"layers": [], "weights": []}

    for layer in model.net:
        if isinstance(layer, nn.Linear):
            model_dict["layers"].append(
                {"in_features": layer.in_features, "out_features": layer.out_features}
            )
            model_dict["weights"].append(
                {
                    "weight": layer.weight.detach().tolist(),
                    "bias": layer.bias.detach().tolist(),
                }
            )
    return model_dict


def dict_to_model(model_json):
    """
    Rebuilds DynamicMLP from JSON weights and layers
    """
    layers_info = model_json["layers"]
    categories = model_json["categories"]
    num_categories = len(categories)

    # Extract hidden layers from JSON (exclude last output layer)
    hidden_layers = [layer["out_features"] for layer in layers_info[:-1]]

    input_dim = layers_info[0]["in_features"]

    # Build model
    model = DynamicMLP(input_dim, hidden_layers, num_categories)

    # Load weights
    weight_layers = [layer for layer in model.net if isinstance(layer, nn.Linear)]
    for idx, layer in enumerate(weight_layers):
        w = torch.tensor(model_json["weights"][idx]["weight"])
        b = torch.tensor(model_json["weights"][idx]["bias"])
        layer.weight.data = w.float()
        layer.bias.data = b.float()

    return model


def labels_to_vector(label_names, categories):
    """
    Converts a label name or list of names to a multi-hot vector.
    """
    if isinstance(label_names, str):
        label_names = [label_names]

    vector = [0] * len(categories)
    for name in label_names:
        if name not in categories:
            raise ValueError(f"Label '{name}' not in model categories: {categories}")
        idx = categories.index(name)
        vector[idx] = 1
    return vector
