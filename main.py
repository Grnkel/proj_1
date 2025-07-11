from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
import numpy as np
import cv2

# TODO skapa klass för skiten
# TODO behöver kind of motsatsen, börja med en form och få antal slices + fit
# TODO använd insikter från convolutional networks, kolla lite diskret optimering
# TODO nu börjar det verkligen bli jobbigt att inte ha en klass för detta
# TODO legit kolla diskret optimering för att få fram nice rekommenderade dimensioner

def extend(im, x=0, y=0):
    assert(x>=0)
    assert(y>=0)

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

def fit_slices(im, v_slices=1, h_slices=1):
    assert(v_slices>0)
    assert(h_slices>0)

    im_height, im_width = np.shape(im)
    ch_height = int(np.ceil(im_height / v_slices))
    ch_width = int(np.ceil(im_width / h_slices))
    
    v_diff = ch_height * v_slices - np.shape(im)[0]
    h_diff = ch_width * h_slices - np.shape(im)[1]
    
    r_v_slice = int(np.ceil(im_height / ch_height))
    r_h_slice = int(np.ceil(im_width / ch_width))

    print("recommended slices:", "v:", r_v_slice, "h:", r_h_slice)
    print("diffs:", v_diff, h_diff)
    print("chunk dims:", ch_height, ch_width)
    
    return extend(im, v_diff, h_diff), ch_height, ch_width

def fit_chunk(im, width, height):

    im_height,im_width = np.shape(im)
    v_slices = int(np.ceil(im_height / height))
    h_slices = int(np.ceil(im_width / width))

    v_diff = height * v_slices - im_height
    h_diff = width * h_slices - im_width

    r_height = int(np.ceil(im_height / v_slices))
    r_width = int(np.ceil(im_width / h_slices))

    print("recommended dims:", "w:", r_width, "h:", r_height)
    print("slices:", "v:", v_slices, "h:", h_slices)
    print("diffs:", v_diff, h_diff)

    return extend(im, v_diff, h_diff), v_slices, h_slices

def pararell(shm_name, shape, dtype, cores, v_slices, h_slices, height_ch, width_ch, dual):
    i, j = dual
    shm = shared_memory.SharedMemory(name=shm_name)
    image = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    v_start = int(np.ceil(i))
    v_end = int(np.ceil((i + 1))) 
    h_start = int(np.ceil(j * h_slices / cores))
    h_end = int(np.ceil((j + 1) * h_slices / cores)) 

    for row in range(v_slices)[v_start:v_end]:
        for col in range(h_slices)[h_start:h_end]: 
            ROW = slice(row * height_ch, (row + 1) * height_ch)
            COL = slice(col * width_ch, (col + 1) * width_ch)

            image = contrast(image, 20, 0.5, ROW,COL, height_ch, width_ch)            
    shm.close()

def contrast(im, k, hw, row, col, h_ch, w_ch):
    x = np.sum(im[row, col])/(h_ch*w_ch) / 255
    exp = np.e**(k*(x-hw))
    sigmoid = exp / (1 + exp)
    im[row, col] = sigmoid * 255
    return im

def main():
    IMAGE = cv2.imread('images/image1.jpg')
    IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2GRAY)
    CH_HEIGHT = 10 
    CH_WIDTH = 10
    CORES = cpu_count()
    IMAGE, horizontal_slices, vertical_slices = fit_chunk(IMAGE, CH_HEIGHT, CH_WIDTH)
    
    shm = shared_memory.SharedMemory(create=True, size=IMAGE.nbytes)
    shared_mem_image = np.ndarray(IMAGE.shape, dtype=IMAGE.dtype, buffer=shm.buf)
    shared_mem_image[:] = IMAGE[:]

    partial_func = partial(
            pararell,
            shm.name,
            IMAGE.shape,
            IMAGE.dtype,
            CORES,
            vertical_slices,
            horizontal_slices,
            CH_HEIGHT,
            CH_WIDTH
        )
    space = np.array([
        (row, col)
        for row in range(vertical_slices)
        for col in range(CORES)
    ])
    with Pool(processes=cpu_count()) as pool:
        pool.map(partial_func, space)

    cv2.imshow('', shared_mem_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    shm.close()
    shm.unlink()

main()