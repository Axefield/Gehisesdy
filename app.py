from flask import Flask, request, render_template, send_file
import os
from pixel_sort_beta import apply_pixel_sorting, process_audio
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Ensure upload directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('static', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get image file
        file = request.files['file']
        if not file:
            return 'No file uploaded', 400

        # Save uploaded image
        image_path = os.path.join('uploads', 'input.png')
        file.save(image_path)

        # Get parameters
        sigma1 = float(request.form.get('sigma1', 8))
        sigma2 = float(request.form.get('sigma2', 2))
        num_sorted = int(request.form.get('num_sorted', 100000))

        # Process audio if provided
        audio_data = None
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file:
                audio_path = os.path.join('uploads', 'audio.wav')
                audio_file.save(audio_path)
                
                min_freq = float(request.form.get('min_freq', 0))
                max_freq = float(request.form.get('max_freq', 150))
                sensitivity = float(request.form.get('sensitivity', 5))
                
                audio_data = process_audio(audio_path, min_freq, max_freq, sensitivity)

        # Apply pixel sorting effect
        result = apply_pixel_sorting(image_path, sigma1, sigma2, num_sorted, audio_data)

        # Convert numpy array to PIL Image
        result_image = Image.fromarray(result)

        # Save to bytes buffer
        img_io = io.BytesIO()
        result_image.save(img_io, 'PNG')
        img_io.seek(0)

        # Clean up temporary files
        try:
            os.remove(image_path)
            if audio_data is not None:
                os.remove(audio_path)
        except:
            pass

        return send_file(img_io, mimetype='image/png')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 