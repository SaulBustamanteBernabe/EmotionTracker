from tensorflow.keras.models import load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# 1. Base del modelo (preentrenado en DeepFace)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))  # también podrías usar Facenet, ArcFace, etc.

# 2. Congelamos capas base
for layer in base_model.layers:
    layer.trainable = False

# 3. Añadimos nuevas capas de clasificación
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
preds = Dense(5, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# 4. Dataset con tus imágenes (ejemplo estilo FER2013)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    "dataset/",
    target_size=(224,224),  # tamaño para VGG-Face
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "dataset/",
    target_size=(224,224),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)

# 5. Entrenamiento
history = model.fit(train_data, validation_data=val_data, epochs=40)

# 6. Guardar el modelo
model.save("my_emotion_model.h5")
