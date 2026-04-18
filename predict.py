import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

# 1. Suppress TensorFlow logging for a cleaner terminal output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 2. Define the exact same classes from training (alphabetic order from your folders)
class_names = [
    'freshapples', 'freshbanana', 'freshcucumber', 'freshokra', 
    'freshoranges', 'freshpotato', 'freshtomato',
    'rottenapples', 'rottenbanana', 'rottencucumber', 'rottenokra', 
    'rottenoranges', 'rottenpotato', 'rottentomato'
]

# 3. Load your trained model
print("Loading model... (This might take a few seconds)")
model = tf.keras.models.load_model('freshness_model.keras')

# 4. Create the prediction function
def predict_freshness(img_path):
    try:
        # Load and resize image to match MobileNetV2 input
        img = image.load_img(img_path, target_size=(224, 224))
        
        # Convert to array and add a batch dimension (models expect batches, not single images)
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        # Make the prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Extract the highest probability
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = 100 * np.max(predictions[0])
        
        print("\n" + "="*30)
        print(f"  PREDICTION RESULTS")
        print("="*30)
        print(f"File:       {img_path}")
        print(f"Class:      {predicted_class.upper()}")
        print(f"Confidence: {confidence:.2f}%")
        print("="*30 + "\n")
        
    except FileNotFoundError:
        print(f"\nError: Could not find the image '{img_path}'. Please check the path.\n")

# Run via terminal
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python predict.py <path_to_image>")
        print("Example: python predict.py dataset/Test/freshbanana/Screen_Shot_2018-06-12_at_9.48.05_PM.png\n")
    else:
        target_image = sys.argv[1]
        predict_freshness(target_image)