import numpy as np
import cv2

image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

height, width = np.shape(image)

vertical_slices = 4
horizontal_slices = 3

chunk_height = height / vertical_slices
chunk_width = width / horizontal_slices

def extend(image, x=0, y=0):
    height, width = np.shape(image)
    for x in range(x):
        i = 0 if x % 2 == 0 else -1
        image = np.insert(image,i,image[:,i],axis=1)
    for y in range(y):
        i = 0 if y % 2 == 0 else -1
        image = np.insert(image,i,image[i,:],axis=0)
    return image

#print(arr)
print(extend(arr,2))
image = extend(image, 10,10)

#image[1:100, 1:100] = np.clip(image[1:100, 1:100] * 0, 0, 255)



"""
def cut(image, x_slices = 1, y_slices = 1):
    height, width = np.shape(image)
    if height % 2 != 0:
        image[:,:]
"""




 
cv2.imshow('', image)
cv2.waitKey(0)
cv2.destroyAllWindows()