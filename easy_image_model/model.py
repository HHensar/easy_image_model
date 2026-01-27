import os
import random

import torch
import torch.nn as nn

from .utils import (
    DynamicMLP,
    dict_to_model,
    labels_to_vector,
    model_to_dict,
    preprocess_image,
)


def create_model(layers, categories, img_size=224, channels=3):
    if (
        not isinstance(layers, list)
        or len(layers) == 0
        or not all(isinstance(x, int) and x > 0 for x in layers)
    ):
        raise ValueError("layers must be a list of positive integers")
    if not isinstance(categories, list) or not all(
        isinstance(x, str) for x in categories
    ):
        raise ValueError("categories must be a list of strings")
    if len(categories) == 0:
        raise ValueError("categories cannot be empty")
    if len(set(categories)) != len(categories):
        raise ValueError("categories must be unique")
    if img_size <= 0:
        raise ValueError("img_size must be positive")

    input_dim = img_size * img_size * channels
    num_categories = len(categories)

    model = DynamicMLP(input_dim, layers, num_categories)
    model_dict = model_to_dict(model)
    model_dict["categories"] = categories  # save categories in JSON
    model_dict["img_size"] = img_size  # save image size for consistency
    model_dict["channels"] = channels
    return model_dict


def train_model(model_json, img_path, labels, lr=1e-4):
    if not isinstance(model_json, dict) or "categories" not in model_json:
        raise ValueError("Invalid model_json")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    img_size = model_json.get("img_size", 224)

    img_tensor = preprocess_image(img_path, img_size)  # [1, input_dim]

    label_vector = labels_to_vector(labels, model_json["categories"])
    labels_tensor = torch.tensor([label_vector], dtype=torch.float)

    model = dict_to_model(model_json)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    logits = model(img_tensor)
    loss = loss_fn(logits, labels_tensor)
    loss.backward()
    optimizer.step()

    updated_json = model_to_dict(model)
    updated_json["categories"] = model_json["categories"]
    updated_json["img_size"] = img_size
    updated_json["channels"] = model_json.get("channels", 3)

    return updated_json


def train_model_batch_folders(model_json, folder_paths, batch_size=4, epochs=5, lr=1e-4):
    if not isinstance(model_json, dict) or "categories" not in model_json:
        raise ValueError("Invalid model_json")
    if not isinstance(folder_paths, dict):
        raise ValueError("folder_paths must be a dict")
    if batch_size <= 0 or epochs <= 0 or lr <= 0:
        raise ValueError("batch_size, epochs, lr must be positive")

    categories = model_json["categories"]
    if set(folder_paths.keys()) != set(categories):
        raise ValueError("folder_paths keys must match model categories")

    for label, path in folder_paths.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Folder not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and move to device
    model = dict_to_model(model_json)
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    all_data = []
    categories = model_json["categories"]
    img_size = model_json.get("img_size", 224)

    for label, folder_path in folder_paths.items():
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img_tensor = preprocess_image(img_path, img_size)  # [1, input_dim]
            label_tensor = torch.tensor(
                [labels_to_vector(label, categories)], dtype=torch.float
            )
            all_data.append((img_tensor, label_tensor))

    for epoch in range(epochs):
        random.shuffle(all_data)

        for i in range(0, len(all_data), batch_size):
            batch = all_data[i : i + batch_size]

            # Stack images and labels
            batch_imgs = torch.cat([x[0] for x in batch], dim=0).to(device)
            batch_labels = torch.cat([x[1] for x in batch], dim=0).to(device)

            optimizer.zero_grad()
            logits = model(batch_imgs)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} complete, last batch loss: {loss.item():.4f}")

    updated_json = model_to_dict(model)
    updated_json["categories"] = categories
    updated_json["img_size"] = img_size
    updated_json["channels"] = model_json.get("channels", 3)

    return updated_json


def evaluate_model(model_json, img_path):
    if not isinstance(model_json, dict) or "categories" not in model_json:
        raise ValueError("Invalid model_json")
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    img_size = model_json.get("img_size", 224)  # backward compatibility

    img_tensor = preprocess_image(img_path, img_size)  # shape [1, input_dim]

    model = dict_to_model(model_json)
    model.eval()

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)[0]  # shape [num_categories]

    categories = model_json["categories"]
    result = {name: float(prob) for name, prob in zip(categories, probs)}

    return result
