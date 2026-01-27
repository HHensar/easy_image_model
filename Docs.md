# Easy Image Model Documentation

This document describes the main functions in the `easy_image_model` package for creating, training, and evaluating image classification models.

## Functions

### create_model(layers, categories, img_size=224, channels=3)

Creates a new image classification model.

**Parameters:**
- `layers` (list of int): List of positive integers specifying the hidden layer sizes.
- `categories` (list of str): List of unique category names for classification.
- `img_size` (int, optional): Image size (default 224). Must be positive.
- `channels` (int, optional): Number of image channels (default 3).

**Returns:**
- dict: Model configuration as a dictionary, including categories, img_size, and channels.

**Raises:**
- ValueError: If layers or categories are invalid.

### train_model(model_json, img_path, labels, lr=1e-4)

Trains the model on a single image with given labels.

**Parameters:**
- `model_json` (dict): Model configuration from `create_model`.
- `img_path` (str): Path to the image file.
- `labels` (list of str): List of category labels for the image.
- `lr` (float, optional): Learning rate (default 1e-4).

**Returns:**
- dict: Updated model configuration after training.

**Raises:**
- ValueError: If model_json is invalid.
- FileNotFoundError: If img_path does not exist.

### train_model_batch_folders(model_json, folder_paths, batch_size=4, epochs=5, lr=1e-4)

Trains the model on batches of images from folders, where each folder corresponds to a category.

**Parameters:**
- `model_json` (dict): Model configuration from `create_model`.
- `folder_paths` (dict): Dictionary mapping category names to folder paths containing images.
- `batch_size` (int, optional): Number of images per batch (default 4).
- `epochs` (int, optional): Number of training epochs (default 5).
- `lr` (float, optional): Learning rate (default 1e-4).

**Returns:**
- dict: Updated model configuration after training.

**Raises:**
- ValueError: If inputs are invalid.
- FileNotFoundError: If folders do not exist.

### evaluate_model(model_json, img_path)

Evaluates the model on a single image and returns classification probabilities.

**Parameters:**
- `model_json` (dict): Model configuration.
- `img_path` (str): Path to the image file.

**Returns:**
- dict: Dictionary mapping category names to probabilities.

**Raises:**
- ValueError: If model_json is invalid.
- FileNotFoundError: If img_path does not exist.
