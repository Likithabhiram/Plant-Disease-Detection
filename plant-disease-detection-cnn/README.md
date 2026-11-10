# Plant Disease Detection Using CNN

A complete starter project that trains a CNN on a folder-based plant leaf image dataset and serves predictions through a Flask web app.

## 1. Project structure

```text
plant-disease-detection-cnn/
├── app.py
├── train.py
├── split_dataset.py
├── config.py
├── requirements.txt
├── .gitignore
├── README.md
├── dataset/
│   └── raw/                 # Put class folders here (not included)
├── models/                  # Created after training
├── static/
│   ├── css/style.css
│   └── uploads/              # Uploaded images
└── templates/
    ├── index.html
    └── result.html
```

## 2. Dataset

Download the PlantVillage dataset (for example, from Kaggle):
https://www.kaggle.com/datasets/emmarex/plantdisease

Extract it so each class has its own folder under `dataset/raw/`. Example:

```text
dataset/raw/
├── Tomato___Early_blight/
│   ├── image1.jpg
│   └── image2.jpg
├── Tomato___healthy/
├── Potato___Early_blight/
└── ...
```

Use the original class folder names. This project discovers the classes automatically, so it is not restricted to 11 classes.

## 3. Setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If TensorFlow installation is not supported by your local Python/OS combination, use Google Colab or a supported Python environment.

## 4. Split the dataset

```bash
python split_dataset.py
```

This creates a reproducible 70% train / 15% validation / 15% test split:

```text
dataset/
├── raw/
├── train/
├── val/
└── test/
```

Do not manually put the split folders inside `dataset/raw`.

## 5. Train the CNN

```bash
python train.py
```

The script:
- Loads images from `dataset/train`, `dataset/val`, and `dataset/test`.
- Resizes images to 224 x 224.
- Applies augmentation to training images only.
- Trains a CNN with dropout and early stopping.
- Saves the model to `models/plant_disease_model.keras`.
- Saves class labels to `models/class_names.json`.
- Saves training curves and a test classification report.

Training can take time. Google Colab with GPU acceleration is a good option.

## 6. Run the web app

After training completes:

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser. Upload a clear leaf image to get a predicted class and model confidence.

## 7. Important notes

- This is an image-classification demonstration, not a certified agricultural diagnostic tool.
- PlantVillage images often have clean backgrounds. Test with real field photographs before making claims about real-world performance.
- The displayed confidence is the model's softmax score; it is not a guarantee that the prediction is correct.
- Use a dataset split that avoids near-duplicate or same-source images crossing train/test sets where possible.
- Do not commit the dataset, trained model, or user-uploaded images to GitHub unless you have checked dataset licensing and repository size. `.gitignore` excludes these by default.

## 8. Suggested GitHub description

CNN-based plant leaf disease classification using TensorFlow/Keras, with image preprocessing, data augmentation, model evaluation, and a Flask prediction interface.
