import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Зареждане на модела
model = load_model("flower_model.keras")

# Списък с класове (трябва да са същите като в dataset-а)
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Път до тестово изображение
img_path = "rose_2.png"  # сложи тук твоя снимка

# Зареждане и обработка
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Предсказване
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Резултат
print("🌸 Това е:", class_names[np.argmax(score)])
print("📊 Сигурност:", round(100 * np.max(score), 2), "%")