import os
import numpy as np
import requests
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)
print("Initializing Flask app...")

# Load the TensorFlow model
model_path = '../saved_model 2/'  # Replace with the actual path to your .pb file
print(f"Loading model from: {model_path}")
model = tf.saved_model.load(model_path)
print("Model loaded successfully.")

headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "http://google.com",  # Referer URL for context (optional, can be the origin page)
}

# Function to process image URL and run inference on the model
def process_image(image_url):
    print(f"Processing image from URL: {image_url}")

    try:
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching image: {e}")
        raise

    # Just warn if it's not JPEG
    content_type = response.headers.get("Content-Type", "").lower()
    if "jpeg" not in content_type and "jpg" not in content_type:
        print(f"⚠️  Warning: image from {image_url} is not a JPEG (Content-Type: {content_type})")

    # Convert to RGB to remove alpha or handle grayscale
    img = Image.open(BytesIO(response.content)).convert("RGB")

    # Resize, normalize, cast
    print("Preprocessing image...")
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Run model
    print("Running inference...")
    infer = model.signatures['serving_default']
    output = infer(tf.convert_to_tensor(img))

    output_key = list(output.keys())[0]
    print(f"Inference complete. Output key: {output_key}")

    return output[output_key].numpy()

@app.after_request
def after_request(response):
    # Set CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'  # Allow all domains (adjust for production)
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'  # Allowed methods
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'  # Allowed headers (adjust as necessary)
    return response

@app.route('/process_images', methods=['POST'])
def process_images():
    print("Received request at /process_images endpoint.")
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON."}), 400
    
    if 'urls' not in data:
        return jsonify({"error": "No 'urls' field found in request."}), 400
    
    image_urls = data['urls']
    
    if not isinstance(image_urls, list):
        return jsonify({"error": "'urls' must be a list of image URLs."}), 400
    
    print(f"Processing {len(image_urls)} image(s)...")
    results = []
    for url in image_urls:
        try:
            result = process_image(url)
            results.append(result.tolist())
        except Exception as e:
            print(f"Error processing image: {e}")
            results.append({"error": str(e), "url": url})

    print("All images processed. Sending response.")
    return jsonify(results)

if __name__ == '__main__':
    print("Flask server starting... Ready to receive requests!")
    app.run(debug=True)

