import onnxruntime as ort
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_image(img):
    """
    Converts image to numpy array and preprocesses it.
    Based on homework 8, uses ImageNet normalization.
    """
    # Convert to numpy array
    x = np.array(img, dtype='float32')
    
    # Rescale from [0, 255] to [0, 1]
    x = x / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    x = (x - mean) / std
    
    # Transpose from (height, width, channels) to (channels, height, width)
    x = x.transpose(2, 0, 1)
    
    # Add batch dimension
    x = np.expand_dims(x, axis=0)
    
    return x

# Question 1: Get output node name
session = ort.InferenceSession('hair_classifier_v1.onnx')
print("Question 1 - Output node name:")
print(f"Output: {session.get_outputs()[0].name}")
print()

# Question 2: Target size
print("Question 2 - Target size: 200x200")
print()

# Question 3: First pixel R channel value after preprocessing
url = "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg"
img = download_image(url)
img_prepared = prepare_image(img, (200, 200))
x = preprocess_image(img_prepared)

print("Question 3 - First pixel R channel value:")
print(f"Value: {x[0, 0, 0, 0]:.3f}")
print()

# Question 4: Model prediction
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: x})
prediction = output[0][0][0]

print("Question 4 - Model output:")
print(f"Prediction: {prediction:.2f}")
print()

print("\n=== Lambda Function Code ===\n")
print("""
def lambda_handler(event, context):
    url = event['url']
    
    # Download and prepare image
    img = download_image(url)
    img_prepared = prepare_image(img, (200, 200))
    x = preprocess_image(img_prepared)
    
    # Load model and predict
    session = ort.InferenceSession('hair_classifier_empty.onnx')
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: x})
    prediction = float(output[0][0][0])
    
    return {
        'prediction': prediction
    }
""")