

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# 1) Зареди предобучения модел MobileNetV2 (weights върху ImageNet)
model = MobileNetV2(weights='imagenet')
model.trainable = False  # гарантираме инференс

# 2) Дефинирай входните трансформации (размери за MobileNetV2)
# MobileNetV2 очаква входи в размери 224x224, с preprocessing от preprocess_input
def prepare_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # точно 224x224
    x = np.array(img)
    x = np.expand_dims(x, axis=0)  # добави batch dimension
    x = preprocess_input(x)        # нормализация за MobileNetV2
    return x

# 3) Инференс и върни топ-5 класи
def predict_image_tf(image_path, top_k=5):
    x = prepare_image(image_path)
    preds = model.predict(x)  # оригинална numpy array с размер (1, 1000)
    top_indices = preds[0].argsort()[-top_k:][::-1]  # индекси на топ-k
    top_probs = preds[0][top_indices]

    # Декодирай към човешки имена (ако желаеш)
    decoded = decode_predictions(preds, top=top_k)[0]  # списък от tuples (class, name, prob)

    # Връщаме две формати:
    # 1) чисти индекси и вероятности
    results_raw = [{"class_id": int(idx), "probability": float(prob)} for idx, prob in zip(top_indices, top_probs)]
    # 2) човекопонятни имена от decode_predictions
    results_decoded = [{"name": name, "description": desc, "probability": float(prob)}
                       for (_, name, prob), prob in zip(decoded, top_probs)]
    return {"raw": results_raw, "decoded": results_decoded}

# Пример за използване:
if __name__ == "__main__":
    img_path = "path/to/your/image.jpg"  # заменете с реален път
    out = predict_image_tf(img_path, top_k=5)
    print("Top-5 (RAW indices):")
    for r in out["raw"]:
        print(f"Class ID: {r['class_id']}, Probability: {r['probability']:.4f}")
    print("\nTop-5 (Decoded):")
    for d in out["decoded"]:
        print(f"{d['name']} ({d['description']}) - {d['probability']:.4f}")