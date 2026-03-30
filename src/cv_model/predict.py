import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model ONCE when the app starts, not every time we predict
try:
    # Adjust path depending on where you run main.py from
    MODEL_PATH = "models/waste_classifier.h5" 
    model = tf.keras.models.load_model(MODEL_PATH)
except:
    print(f"Error: Could not find model at {MODEL_PATH}. Did you run the training notebook?")
    model = None

# These must match the alphabetical order from your training data!
CLASS_NAMES = ['Glass', 'Metal', 'Paper', 'Plastic']

def classify_waste(image_file):
    """
    Takes an uploaded image or camera frame, processes it, and returns the classification.
    """
    if model is None:
        return "Model not loaded", 0.0

    # 1. Open the image and resize it to what the model expects (224x224)
    img = Image.open(image_file).convert('RGB')
    img = img.resize((224, 224))
    
    # 2. Convert to numpy array and add a batch dimension
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch of 1
    
    # 3. Make the prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) # Get probabilities
    
    # 4. Find the highest probability
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return predicted_class, confidence