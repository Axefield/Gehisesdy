import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import librosa
import scipy
import random

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







image=np.array(plt.imread("nameofyourimage.jpg"))# .jpg image that will be used for editing.
image_edit=image                                 # this will be the edited image, so original copy isn't changed
grayscale_rgb_weights=[0.2126, 0.7152, 0.0722]   # color weights used for manually calculating gray scale.
image_grayscale=(image[:,:,0]*grayscale_rgb_weights[0]+image[:,:,1]*grayscale_rgb_weights[1]+image[:,:,2]*grayscale_rgb_weights[2]).astype(int) #calculating grayscale

sigma1=8 # used for calculating probability map (both need to be positive or 0)
sigma2=2 # sigma1 > sigma2 ~= more attention to edges/big shapes in map, sigma1 < sigma2 ~= more attention to detail 
probability_map=scipy.ndimage.gaussian_filter(image_grayscale,sigma1)-scipy.ndimage.gaussian_filter(image_grayscale,sigma2)
probability_map[probability_map<0]=0
probability_map[probability_map>255]=255

#plt.imshow(probability_map,cmap='gray') This plot will show the probabilty map
#plt.show()








###############################################################################################
# this section is just for extra info on probability map, not needed but is kind of neat. may be helpful for sorting out points

#percentiles=np.percentile(probability_map,[0,10,20,30,40,50,60,70,80,90,100])
#print('0th, 10th, 20th... percentiles')
#print(str(percentiles[0])+" "+str(percentiles[1])+" "+str(percentiles[2])+" "+str(percentiles[3])+" "+str(percentiles[4])+" "+str(percentiles[5])+" "+str(percentiles[6])+" "+str(percentiles[7])+" "+str(percentiles[8])+" "+str(percentiles[9])+" "+str(percentiles[10]))

###############################################################################################






num_sorted=100000 # tries to get this many pixel sorting points
max_iterations=10000000 # if it can not get the desired amount of pixel sorting points, for loop will terminated after this many runs

pixel_sort_location=np.zeros((image.shape[0],image.shape[1],image.shape[2])) # this matrix is a log of points that will be used for pixel sorting
pixel_sort_counter=0                      # this keeps track of how many points you currently have for sorting in the for loop
normal_sort_distance=image.shape[0]*0.03  # this is the mean length of pixel sorting, as a percentage of vertical pixels in the image
sigma_sort_distance=image.shape[0]*0.01   # this is the standard deviation of the pixel sorting length, as a percentage of the vertical pixels in the image

for i in range(max_iterations):
    d0=random.randint(0,image_grayscale.shape[0]-1) # d0 and d1 are a random point to test on probability map
    d1=random.randint(0,image_grayscale.shape[1]-1)
    value=random.randint(1,255)                     # if value <= probability_map[d0,d1], point [d0,d1] will be logged to be sorted
    if probability_map[d0,d1]>=value:
        if pixel_sort_location[d0,d1,0]!=1:         # this makes sure a point isn't logged twice
            pixel_sort_location[d0,d1,:]=1          # logging the point for sorting at a later time
            pixel_sort_counter=pixel_sort_counter+1 # counting how many sortings points are currently had
    if pixel_sort_counter==num_sorted:              # breaks the loop when the goal amount of sorting points is met
        pixel_sort_counter=0
        print(str(num_sorted)+' sorted')
        break
    if i==max_iterations-1:                         # this the end of the for loop when the max iteration limit is met, tells you how many points were selected
        print(str(pixel_sort_counter)+' sorted')
        pixel_sort_counter=0
    

for i in range(image.shape[0]):                     # nested for loop goes through every point in the image to see if it was logged for sorting
    for j in range(image.shape[1]):                 # this is where the sorting happens
        if pixel_sort_location[i,j,0]==1:           # this if states checks if a point was logged
            sort_distance=np.array(random.gauss(normal_sort_distance,sigma_sort_distance)).astype(int) # randomly calculate the distance of each pixel sort on a normal distrobution
            end_sort_point=i+sort_distance          # end point for the sort
            if end_sort_point>=image.shape[0]:      # these two if statements make sure the end point is within the image
                end_sort_point=image.shape[0]-1
            if end_sort_point<0:
                end_sort_point=0
            for k in range(i,end_sort_point):       # this for loop goes through every pixel along a sort
                    image_edit[k,j,:]=((image[i,j,:]+image[k,j,:])/2).astype(int) # generates the color for each pixel along the sort

###################################################################################################################################################
# this part of the script just saves the edited image to your computer
                
#output_hor=                                   # your output wont be the same resolution as your input first time running the script. if you want it to be the original resolution,
#scale_factor=image.shape[1]/output_hor        # after you run the script the first time, make output_hor the horizontal dimension of your output image, un comment output_hor and
                                               # also scale_factor. then mutiply scale_factor by the dpi value in plt.savefig
                                               # it still may not be exactly the same dimension but it will be really close

plt.axis('off')               
plt.imshow(image_edit/255)
plt.savefig("nameofyoureditedimage.jpg", bbox_inches='tight',pad_inches=0,dpi=100) # this line is what actually save the edited image
plt.show()
                
            
        


    
        



