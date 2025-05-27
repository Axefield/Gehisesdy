import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import librosa
import scipy
import random
from numba import jit

#######################################################################################

# for audio Reactivity
# if you want to mess with this feel free to ask question
# but the rest of the sorting script will need to be heavily edited for this to work

#file_name=''
#sound_data, sr=librosa.load(file_name, sr=None)
#durration_audio=np.size(sound_data)/sr
#fps=25
#total_frames=durration_audio*fps
#sr20fps=np.size(sound_data)/(total_frames)
#f,t,Zxx=scipy.signal.stft(sound_data, fs=sr, nperseg=(np.size(sound_data)/durration_audio)/fps,noverlap=0)


#max_freq=150
#for i in range(np.size(f)):
#    if f[i]>=max_freq:
#        max_freq_index=i
#        break
    
    
#min_freq=0
#for i in range(np.size(f))[::-1]:
#    if f[i]<=min_freq:
#        min_freq_index=i
#        break

#avg_strength=[]
#for i in range(np.size(t)):
#    avg_strength.append(np.abs(Zxx[min_freq_index:max_freq_index,i]).mean())
    
#avg_strength=avg_strength/max(avg_strength)

#######################################################################################

@jit(nopython=True)
def apply_sorting_effect(image_edit, image, pixel_sort_location, normal_sort_distance, sigma_sort_distance, audio_resampled=None):
    height, width = image.shape[:2]
    for i in range(height):
        for j in range(width):
            if pixel_sort_location[i, j, 0] == 1:
                if audio_resampled is not None:
                    sort_distance = int(np.random.normal(
                        normal_sort_distance * (1 + audio_resampled[i]),
                        sigma_sort_distance
                    ))
                else:
                    sort_distance = int(np.random.normal(
                        normal_sort_distance,
                        sigma_sort_distance
                    ))
                
                end_sort_point = min(i + sort_distance, height - 1)
                if end_sort_point > i:
                    image_edit[i:end_sort_point, j] = np.mean(
                        np.stack([image[i, j], image[i:end_sort_point, j]]),
                        axis=0
                    ).astype(np.uint8)
    return image_edit

def process_audio(audio_path, min_freq=0, max_freq=150, sensitivity=5):
    # Load audio file with reduced sample rate for better performance
    sound_data, sr = librosa.load(audio_path, sr=22050)
    
    # Calculate STFT with optimized parameters
    f, t, Zxx = scipy.signal.stft(
        sound_data,
        fs=sr,
        nperseg=1024,
        noverlap=512
    )
    
    # Vectorized frequency range selection
    freq_mask = (f >= min_freq) & (f <= max_freq)
    if not np.any(freq_mask):
        return np.ones(len(t))  # Return neutral data if no frequencies in range
    
    # Vectorized strength calculation
    avg_strength = np.abs(Zxx[freq_mask]).mean(axis=0)
    
    # Normalize and apply sensitivity
    avg_strength = (avg_strength / avg_strength.max()) ** (1 / sensitivity)
    
    return avg_strength

def apply_pixel_sorting(image_path, sigma1=8, sigma2=2, num_sorted=100000, audio_data=None):
    # Check file format
    file_ext = image_path.lower().split('.')[-1]
    if file_ext not in ['png', 'jpg', 'jpeg']:
        raise ValueError("Unsupported image format. Please use PNG or JPEG files.")
    
    # Load and preprocess image
    image = plt.imread(image_path)
    
    # Handle color ranges
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    
    # Ensure correct number of channels based on format
    if file_ext == 'png' and image.shape[2] == 3:
        # Convert RGB PNG to RGBA
        alpha = np.full((image.shape[0], image.shape[1], 1), 255, dtype=np.uint8)
        image = np.concatenate([image, alpha], axis=2)
    elif file_ext in ['jpg', 'jpeg'] and image.shape[2] == 4:
        # Convert RGBA JPEG to RGB
        image = image[:, :, :3]
    
    # Create a copy for editing
    image_edit = image.copy()
    
    # Convert to grayscale using vectorized operations
    grayscale_rgb_weights = np.array([0.2126, 0.7152, 0.0722])
    image_grayscale = np.sum(image[:, :, :3] * grayscale_rgb_weights, axis=2).astype(np.uint8)
    
    # Calculate probability map using optimized gaussian filters
    probability_map = scipy.ndimage.gaussian_filter(image_grayscale, sigma1) - \
                     scipy.ndimage.gaussian_filter(image_grayscale, sigma2)
    probability_map = np.clip(probability_map, 0, 255)
    
    # Process audio data if provided
    audio_resampled = None
    if audio_data is not None:
        audio_resampled = scipy.signal.resample(audio_data, image.shape[0])
        audio_resampled = (audio_resampled - audio_resampled.min()) / (audio_resampled.max() - audio_resampled.min())
        probability_map = np.clip(probability_map * (1 + audio_resampled[:, np.newaxis]), 0, 255)
    
    # Generate random points more efficiently
    height, width = image_grayscale.shape
    total_pixels = height * width
    num_points = min(num_sorted, total_pixels)
    
    # Use numpy's random choice for better performance
    indices = np.random.choice(
        total_pixels,
        size=num_points,
        p=probability_map.flatten() / probability_map.sum()
    )
    
    # Convert indices to 2D coordinates
    rows = indices // width
    cols = indices % width
    
    # Create pixel sort location array with matching channels
    pixel_sort_location = np.zeros((height, width, image.shape[2]), dtype=np.uint8)
    pixel_sort_location[rows, cols] = 1
    
    # Calculate sort distances
    normal_sort_distance = height * 0.03
    sigma_sort_distance = height * 0.01
    
    # Apply sorting effect using optimized function
    image_edit = apply_sorting_effect(
        image_edit,
        image,
        pixel_sort_location,
        normal_sort_distance,
        sigma_sort_distance,
        audio_resampled
    )
    
    return image_edit

# Example usage
# if __name__ == '__main__':
#     image_path = "nameofyourimage.jpg"
#     image_edit = apply_pixel_sorting(image_path)
#     plt.axis('off')
#     plt.imshow(image_edit / 255)
#     plt.savefig("nameofyoureditedimage.jpg", bbox_inches='tight', pad_inches=0, dpi=100)
#     plt.show()

###############################################################################################
# this section is just for extra info on probability map, not needed but is kind of neat. may be helpful for sorting out points

#percentiles=np.percentile(probability_map,[0,10,20,30,40,50,60,70,80,90,100])
#print('0th, 10th, 20th... percentiles')
#print(str(percentiles[0])+" "+str(percentiles[1])+" "+str(percentiles[2])+" "+str(percentiles[3])+" "+str(percentiles[4])+" "+str(percentiles[5])+" "+str(percentiles[6])+" "+str(percentiles[7])+" "+str(percentiles[8])+" "+str(percentiles[9])+" "+str(percentiles[10]))

###############################################################################################








###################################################################################################################################################
# this part of the script just saves the edited image to your computer
                
#output_hor=                                   # your output wont be the same resolution as your input first time running the script. if you want it to be the original resolution,
#scale_factor=image.shape[1]/output_hor        # after you run the script the first time, make output_hor the horizontal dimension of your output image, un comment output_hor and
                                               # also scale_factor. then mutiply scale_factor by the dpi value in plt.savefig
                                               # it still may not be exactly the same dimension but it will be really close

# plt.imshow(image_edit/255)

plt.axis('off')               
# plt.imshow(image_edit/255)
plt.savefig("nameofyoureditedimage.jpg", bbox_inches='tight',pad_inches=0,dpi=100) # this line is what actually save the edited image








