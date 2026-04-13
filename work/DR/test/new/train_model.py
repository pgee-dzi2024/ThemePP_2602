import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Път до dataset-а (папката "flowers")
data_dir = "flowers"

# Зареждане на данните
train_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

val_data = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

class_names = train_data.class_names
print("Класове:", class_names)

# Подобряване на производителността
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# Зареждане на готов модел
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # НЕ обучаваме целия модел

# Добавяне на класификатор
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(len(class_names), activation='softmax')(x)

model = keras.Model(inputs=base_model.input, outputs=output)

# Компилиране
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Обучение
model.fit(train_data, validation_data=val_data, epochs=5)

# Запазване
model.save("flower_model.keras")

print("✅ Моделът е запазен като flower_model.keras")