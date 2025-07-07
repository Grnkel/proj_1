import numpy as np
import cv2

image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = np.shape(image)

vertical_slices = 7
horizontal_slices = 7

def extend(im, x=0, y=0):
    for x in range(x):
        i = 0 if x % 2 == 0 else -1
        im = np.insert(im,i,im[:,i],axis=1)
    for y in range(y):
        i = 0 if y % 2 == 0 else -1
        im = np.insert(im,i,im[i,:],axis=0)
    return im

def chunk_shape(im, v_slices=1, h_slices=1):
    assert(v_slices>0)
    assert(h_slices>0)

    height, width = np.shape(im)
    ch_height = np.ceil(height / v_slices)
    ch_width = np.ceil(width / h_slices)
    return ch_height, ch_width

def fit(im, v_slices=1, h_slices=1):
    assert(v_slices>0)
    assert(h_slices>0)

    ch_height, ch_width = chunk_shape(im, v_slices, h_slices)
    v_diff = int(ch_height * v_slices - np.shape(im)[0])
    h_diff = int(ch_width * h_slices - np.shape(im)[1])
    return extend(image, v_diff, h_diff)

#def select(im, v_slice, h_slice): TODO skapa klass för skiten
# TODO behöver kind of motsatsen, börja med en form och få antal slices + fit

image = fit(image, vertical_slices, horizontal_slices)

print(np.shape(image))
 
cv2.imshow('', image)
cv2.waitKey(0)
cv2.destroyAllWindows()