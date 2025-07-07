import numpy as np
import cv2

image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = np.shape(image)

def _extend(image, x=0, y=0):
    for x in range(x):
        i = 0 if x % 2 == 0 else -1
        image = np.insert(image,i,image[:,i],axis=1)
    for y in range(y):
        i = 0 if y % 2 == 0 else -1
        image = np.insert(image,i,image[i,:],axis=0)
    return image

def fit(image, vertical_slices=1, horizontal_slices=1):
    assert(vertical_slices>0)
    assert(horizontal_slices>0)
    
    height, width = np.shape(image)
    chunk_height = np.ceil(height / vertical_slices)
    chunk_width = np.ceil(width / horizontal_slices)
    v_diff = int(chunk_height * vertical_slices - height)
    h_diff = int(chunk_width * horizontal_slices - width)
    return _extend(image, v_diff, h_diff)

image = fit(image, 3, 1)
print(np.shape(image))

#image[1:100, 1:100] = np.clip(image[1:100, 1:100] * 0, 0, 255)






 
cv2.imshow('', image)
cv2.waitKey(0)
cv2.destroyAllWindows()