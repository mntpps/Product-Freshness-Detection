import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os

# 1. Define paths and parameters
train_dir = 'dataset/Train'
test_dir = 'dataset/Test'
BATCH_SIZE = 32
IMG_SIZE = (224, 224) # Standard input size for MobileNetV2
EPOCHS = 10 # Good starting point for transfer learning

# 2. Load the dataset
print("Loading training data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

print("Loading validation/test data...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    shuffle=True, 
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

class_names = train_dataset.class_names
print(f"Classes ({len(class_names)}): {class_names}")

# 3. Optimize datasets for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# 4. Advanced Data Augmentation
# Randomly alters lighting, contrast, and orientation to prevent dataset bias
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip('horizontal'),
  layers.RandomRotation(0.2),
  layers.RandomBrightness(factor=0.3), # Simulates dramatic lighting/shadows
  layers.RandomContrast(factor=0.3),   # Forces the model to ignore deep color variations
])

# 5. Load Pre-trained Base Model (MobileNetV2)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = MobileNetV2(input_shape=IMG_SIZE + (3,),
                         include_top=False, # Crucial: Drop the original 1000-class layer
                         weights='imagenet')

# Freeze the base model so we don't destroy the pre-trained weights
base_model.trainable = False

# 6. Build our custom classification head
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x) # Helps prevent overfitting
# Output layer specifically for our 14 fruit/veg classes
outputs = layers.Dense(len(class_names), activation='softmax')(x) 

model = models.Model(inputs, outputs)

# 7. Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()

# 8. Callbacks
# Automatically saves the best version of your model and stops training early if it stops improving
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('freshness_model.keras', save_best_only=True, monitor='val_accuracy'),
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
]

# 9. Train the model!
print("\nStarting training...")
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset,
    callbacks=callbacks
)

# 10. Save the graphs for your report
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('training_history.png')

print("\nTraining complete! Model saved as 'freshness_model.keras' and graphs saved as 'training_history.png'.")