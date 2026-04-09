from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from django.conf import settings
from PIL import Image
from tensorflow.keras.models import load_model

# Път до обучен модел в приложението main
MODEL_PATH = Path(settings.BASE_DIR) / "main" / "keras_model" / "flower_model_opt.keras"
CLASS_NAMES_PATH = Path(settings.BASE_DIR) / "main" / "keras_model" / "class_names.json"

# Нужна е за десериализация на Lambda слоя в .keras модела
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Превод на български за показване в интерфейса
CLASS_NAMES_BG = {
    "daisy": "маргаритка",
    "dandelion": "глухарче",
    "roses": "рози",
    "sunflowers": "слънчогледи",
    "tulips": "лалета",
}

MODEL = None
CLASS_NAMES_EN = None

def load_resources():
    global MODEL, CLASS_NAMES_EN

    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Моделът не е намерен: {MODEL_PATH}")

        MODEL = load_model(
            MODEL_PATH,
            custom_objects={"preprocess_input": preprocess_input},
            compile=False
        )

    if CLASS_NAMES_EN is None:
        if CLASS_NAMES_PATH.exists():
            with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
                CLASS_NAMES_EN = json.load(f)
        else:
            # fallback, ако json файлът липсва
            CLASS_NAMES_EN = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]


def predict_image(uploaded_file):
    load_resources()

    # Django подава UploadedFile, не file path
    uploaded_file.seek(0)

    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Не прилагаме preprocess_input тук, защото моделът вече го съдържа
    predictions = MODEL.predict(img_array, verbose=0)[0]

    top_index = int(np.argmax(predictions))
    confidence = round(float(predictions[top_index]) * 100, 2)

    class_name_en = CLASS_NAMES_EN[top_index]
    class_name_bg = CLASS_NAMES_BG.get(class_name_en, class_name_en)

    all_predictions = []
    for i, class_name in enumerate(CLASS_NAMES_EN):
        all_predictions.append({
            "class_name_en": class_name,
            "class_name_bg": CLASS_NAMES_BG.get(class_name, class_name),
            "confidence": round(float(predictions[i]) * 100, 2),
        })

    return {
        "predicted_class_en": class_name_en,
        "predicted_class_bg": class_name_bg,
        "confidence": confidence,
        "all_predictions": all_predictions,
    }