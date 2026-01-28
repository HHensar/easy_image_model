# Easy Image Model Documentation

`easy_image_model` is a lightweight Python library for creating, training, and evaluating image classification models with minimal setup.

---

## Functions

### `create_model(layers, categories, img_size=224, channels=3)`

Creates a new image classification model.

#### Parameters
- **`layers`** (`list[int]`)  
  List of positive integers specifying the hidden layer sizes.

- **`categories`** (`list[str]`)  
  List of unique category names for classification.

- **`img_size`** (`int`, optional)  
  Image size (default: `224`). Images are resized to `img_size × img_size`.

- **`channels`** (`int`, optional)  
  Number of image channels (default: `3`).

#### Returns
- **`dict`**  
  Model configuration object used for training and evaluation.

#### Raises
- `ValueError` if layers or categories are invalid.

---

### `train_model(model, img_path, labels, lr=1e-4)`

Trains the model on a single image with the provided labels.

#### Parameters
- **`model`** (`dict`)  
  Model configuration returned by `create_model`.

- **`img_path`** (`str`)  
  Path to the image file.

- **`labels`** (`list[str]`)  
  List of category labels associated with the image.

- **`lr`** (`float`, optional)  
  Learning rate (default: `1e-4`).

#### Returns
- **`dict`**  
  Updated model configuration after training.

#### Raises
- `ValueError` if the model is invalid.
- `FileNotFoundError` if `img_path` does not exist.

---

### `train_model_batch_folders(model, folder_paths, batch_size=4, epochs=5, lr=1e-4)`

Trains the model using batches of images from folders, where each folder represents a category.

#### Parameters
- **`model`** (`dict`)  
  Model configuration returned by `create_model`.

- **`folder_paths`** (`dict[str, str]`)  
  Mapping of category names to folder paths containing images.

- **`batch_size`** (`int`, optional)  
  Number of images per batch (default: `4`).

- **`epochs`** (`int`, optional)  
  Number of training epochs (default: `5`).

- **`lr`** (`float`, optional)  
  Learning rate (default: `1e-4`).

#### Returns
- **`dict`**  
  Updated model configuration after training.

#### Raises
- `ValueError` if inputs are invalid.
- `FileNotFoundError` if any folder paths do not exist.

---

### `evaluate_model(model, img_path)`

Evaluates the model on a single image and returns classification probabilities.

#### Parameters
- **`model`** (`dict`)  
  Trained model configuration.

- **`img_path`** (`str`)  
  Path to the image file.

#### Returns
- **`dict[str, float]`**  
  Mapping of category names to probability scores.

#### Raises
- `ValueError` if the model is invalid.
- `FileNotFoundError` if `img_path` does not exist.
