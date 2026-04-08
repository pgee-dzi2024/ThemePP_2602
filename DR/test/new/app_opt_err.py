import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Зареждане на модела
model = load_model("flower_model_opt_err.keras")

# Класовете трябва да са в ТОЧНИЯ ред на папките
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

print("CLASS NAMES:", class_names)

img_path = "rose_2.png"

# Обработка
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
img_array = np.expand_dims(img_array, axis=0)

predictions = model.predict(img_array, verbose=0)[0]

top3_idx = np.argsort(predictions)[-3:][::-1]

print("\n🌸 ТОП 3 резултата:")
for i in top3_idx:
    print(f"{class_names[i]} → {round(float(predictions[i]) * 100, 2)}%")