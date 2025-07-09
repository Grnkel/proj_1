import numpy as np
import cv2

image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = np.shape(image)

vertical_slices = 50
horizontal_slices = 50

print("the shape of the image:", np.shape(image))

def extend(im, x=0, y=0):
    assert(x>0)
    assert(y>0)

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
    ch_height = int(np.ceil(height / v_slices))
    ch_width = int(np.ceil(width / h_slices))
    return ch_height, ch_width

def fit(im, v_slices=1, h_slices=1):
    assert(v_slices>0)
    assert(h_slices>0)

    ch_height, ch_width = chunk_shape(im, v_slices, h_slices)
    v_diff = ch_height * v_slices - np.shape(im)[0]
    h_diff = ch_width * h_slices - np.shape(im)[1]

    height, width = np.shape(im)
    r_v_slice = int(np.ceil(height / np.ceil(height / v_slices)))
    r_h_slice = int(np.ceil(width / np.ceil(width / h_slices)))

    print("recommended slices:", "v:", r_v_slice, "h:", r_h_slice)
    print("diffs:", v_diff, h_diff)
    return extend(image, v_diff, h_diff)

image = fit(image, vertical_slices, horizontal_slices)
ch_h, ch_w = chunk_shape(image, vertical_slices, horizontal_slices)
for row in range(vertical_slices):
    for col in range(horizontal_slices): 
        ROW = slice(row * ch_h, (row + 1) * ch_h)
        COL = slice(col * ch_w, (col + 1) * ch_w)
        image[ROW, COL] = np.sum(image[ROW, COL]) / (ch_h*ch_w)

        #cv2.imshow('', image)
        #cv2.waitKey(1)

# TODO skapa klass för skiten
# TODO behöver kind of motsatsen, börja med en form och få antal slices + fit
 
cv2.imshow('', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
