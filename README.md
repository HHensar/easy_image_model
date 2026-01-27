# Easy Image Model
### by Harrison Hensarling
**Easy Image Model** is a lightweight Python library for quickly building, training, and evaluating image classification models using PyTorch. 
It is designed for simplicity and ease of use. Supports:

-**small datasets**

-**non-normalized images**

-**batch training**  

---

## Features

- Create a custom neural network with **any number of hidden layers and nodes**.  
- Train models on **folders of labeled images** (each folder represents a category).  
- Batch training for faster convergence.  
- Automatically preprocesses images (resizing, normalization) — **no manual preprocessing required except for labeling**.  
- Evaluate a single image and return **category probabilities**.  
- Save and reload model weights as JSON for portability.  
- **Customizable image input size** (default 224x224, can be adjusted for your dataset).
- Supported image types: 
   - `.jpg` 
   - `.jpeg`
   - `.png`
   - `.bmp`
   - `.gif (only first frame is used)`
   - `.tif`
   - `.tiff`
   - ` More supported but untested`
---

## Installation

```bash
pip install easy-image-model
```

Or for development:

```bash
git clone https://github.com/hhensar/easy-image-model.git
cd easy-image-model
pip install -e ".[dev]"
```

---

## Example Usage

```python
from easy_image_model import create_model, train_model_batch_folders, evaluate_model

# Define categories
categories = ['Eagles', 'Penguins', 'Owls', 'Others']

# Create model with hidden layers (3 layers with 512, 256, 128 nodes)
# img_size defaults to 224, but can be customized (e.g., 128, 256, 512)
model = create_model([512, 256, 128], categories, img_size=224)

# Train on folders (each folder contains images of one category)
folder_paths = {
    'Eagles': 'dataset/Eagles',
    'Penguins': 'dataset/Penguins',
    'Owls': 'dataset/Owls',
    'Others': 'dataset/Others'
}
model = train_model_batch_folders(model, folder_paths, batch_size=4, epochs=5)

# Evaluate a single image
result = evaluate_model(model, 'test_images/test1.jpg')
print(result)
```

**Example Output:**
```
{'Eagles': 0.87, 
 'Penguins': 0.05, 
 'Owls': 0.23, 
 'Others': 0.1}
```