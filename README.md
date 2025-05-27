# Pixel Sorting Glitch Effect

## Overview
This project contains a Python script that applies a pixel sorting glitch effect to images. The effect is achieved by sorting pixels along vertical lines based on a probability map derived from the grayscale version of the image.

## Requirements
- Python 3.6 or higher
- Required libraries (see `requirements.txt`):
  - Pillow
  - OpenCV
  - NumPy
  - Matplotlib
  - Librosa
  - SciPy

## Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place your image in the same directory as the script and name it `nameofyourimage.jpg`.
2. Run the script:
   ```bash
   python pixel_sort_beta.py
   ```
3. The edited image will be saved as `nameofyoureditedimage.jpg`.

## Features
- Converts the input image to grayscale.
- Creates a probability map using Gaussian filters.
- Sorts pixels along vertical lines based on the probability map.
- Saves the edited image.

## Audio Reactivity
The script includes commented-out sections for audio reactivity. If you want to explore this feature, uncomment the relevant code and provide an audio file.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 