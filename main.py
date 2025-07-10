import numpy as np
import cv2
from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial

# TODO skapa klass för skiten
# TODO behöver kind of motsatsen, börja med en form och få antal slices + fit
# TODO använd insikter från convolutional networks, kolla lite diskret optimering
# TODO nu börjar det verkligen bli jobbigt att inte ha en klass för detta

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
    
    return extend(im, v_diff, h_diff)

def pararell_apply(shm_name, shape, dtype, cores, v_slices, h_slices, height_ch, width_ch, i):
    shm = shared_memory.SharedMemory(name=shm_name)
    image = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    start = int(np.ceil(i * v_slices / cores))
    end = int(np.ceil((i + 1) * v_slices / cores)) 

    for row in list(range(v_slices))[start:end]:
        for col in range(h_slices): 
            ROW = slice(row * height_ch, (row + 1) * height_ch)
            COL = slice(col * width_ch, (col + 1) * width_ch)
            image[ROW, COL] = np.sum(image[ROW, COL]) / (height_ch * width_ch)
    shm.close()

def main():
    image = cv2.imread('image.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    vertical_slices = 50
    horizontal_slices = 50

    image = fit(image, vertical_slices, horizontal_slices)
    height_chunk, width_chunk = chunk_shape(image, vertical_slices, horizontal_slices)  

    shm = shared_memory.SharedMemory(create=True, size=image.nbytes)
    shared_mem_image = np.ndarray(image.shape, dtype=image.dtype, buffer=shm.buf)
    shared_mem_image[:] = image[:]

    partial_func = partial(
            pararell_apply,
            shm.name,
            image.shape,
            image.dtype,
            cpu_count(),
            vertical_slices,
            horizontal_slices,
            height_chunk,
            width_chunk
        )

    with Pool(processes=cpu_count()) as pool:
        pool.map(partial_func, range(cpu_count()))

    cv2.imshow('', shared_mem_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    shm.close()
    shm.unlink()

main()