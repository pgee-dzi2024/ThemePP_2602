import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Нужна е само за зареждането на модела
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Зареждане на модела
model = load_model(
    "flower_model_opt.keras",
    custom_objects={"preprocess_input": preprocess_input},
    compile=False
)

# Класовете трябва да са в ТОЧНИЯ ред на папките
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

print("CLASS NAMES:", class_names)

img_path = "img_1.png"

# Зареждане и обработка
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

# ВАЖНО:
# Не прилагаме preprocess_input тук, защото моделът вече го има вътре.
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array, verbose=0)[0]

top3_idx = np.argsort(predictions)[-3:][::-1]

print("\n🌸 ТОП 3 резултата:")
for i in top3_idx:
    print(f"{class_names[i]} → {round(float(predictions[i]) * 100, 2)}%")