from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
import numpy as np
import cv2

import random as rand

class ImageHandler():
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(path)
        self.dims = np.shape(self.image)
        self.chunk_dims = None, None
        self.slices = None, None

    def update(self, image):
        self.image = image
        self.dims = np.shape(self.image)

    def show(self):
        cv2.imshow('', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def grayscale(self):
        self.update(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))

    def extend(self, dy=0, dx=0):
        im = self.image

        # vertical extension or cropping (dy)
        if dy > 0:
            pad_top = dy // 2
            pad_bottom = dy - pad_top
            im = np.pad(im, ((pad_top, pad_bottom), (0, 0)), mode='edge')
        elif dy < 0:
            crop_top = -dy // 2
            crop_bottom = -dy - crop_top
            im = im[crop_top:- crop_bottom, :]

        # horizontal extension or cropping (dx)
        if dx > 0:
            pad_left = dx // 2
            pad_right = dx - pad_left
            im = np.pad(im, ((0, 0), (pad_left, pad_right)), mode='edge')
        elif dx < 0:
            crop_left = -dx // 2
            crop_right = -dx - crop_left
            im = im[:, crop_left:- crop_right]

        self.update(im)

    def fit_slices(self, v_slices=1, h_slices=1):
        assert(v_slices>0)
        assert(h_slices>0)

        # resizing image to fit with slices

        ch_height = -(-self.dims[0] // v_slices)
        ch_width = -(-self.dims[1] // v_slices)

        diff_y = -self.dims[0] % ch_height
        diff_x = -self.dims[1] % ch_width

        self.extend(diff_x,diff_y)

        self.chunk_dims = ch_height, ch_width
        self.slices = v_slices, h_slices

        print("diffs:", diff_y, diff_x)
        print("chunk dims:", ch_height, ch_width)

    def fit_chunk(self, ch_height=1, ch_width=1):
        assert(ch_height>0)
        assert(ch_width>0)

        # resizing image to fit with slices

        v_slices = -(-self.dims[0] // ch_height)
        h_slices = -(-self.dims[1] // ch_width)

        diff_y = ch_height * v_slices - self.dims[0]
        diff_x = ch_width * h_slices - self.dims[1]

        self.extend(diff_x,diff_y)

        self.chunk_dims = ch_height, ch_width
        self.slices = v_slices, h_slices

        print("diffs:", diff_y, diff_x)
        print("slices:", "v:", v_slices, "h:", h_slices)
    
    def apply(self, cores=cpu_count()):
        shm = shared_memory.SharedMemory(create=True, size=self.image.nbytes)
        shared_mem_image = np.ndarray(self.image.shape, dtype=self.image.dtype, buffer=shm.buf)
        shared_mem_image[:] = self.image[:]  # Copy data to shared memory

        partial_func = partial(
            self.parallel,
            shm.name,
            self.image.shape,
            self.image.dtype,
            cores
        )

        space = [
            (i, j)
            for i in range(self.slices[0])
            for j in range(cores)
        ]

        with Pool(processes=cores) as pool:
            pool.map(partial_func, space)

        # Read back into self.image after multiprocessing
        self.image = shared_mem_image.copy()
        shm.close()
        shm.unlink()

    def parallel(self, shm_name, shape, dtype, cores, task):
        i, j = task
        shm = shared_memory.SharedMemory(name=shm_name)
        image = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        v_start = int(np.ceil(i))
        v_end = int(np.ceil((i + 1)))
        h_start = int(np.ceil(j * self.slices[1] / cores))
        h_end = int(np.ceil((j + 1) * self.slices[1] / cores))

        random_core = rand.randrange(15, 240)
        for row in range(v_start, v_end):
            for col in range(h_start, h_end):
                random_chunk = rand.randrange(-15, 15)
                ROW = slice(row * self.chunk_dims[0], (row + 1) * self.chunk_dims[0])
                COL = slice(col * self.chunk_dims[1], (col + 1) * self.chunk_dims[1])
                image[ROW, COL] = random_core + random_chunk

        shm.close()
    
def testing():
    image = ImageHandler('images/image1.jpg')
    image.grayscale()
    image.fit_chunk(5,5)
    image.apply(8)
    
    image.show()

testing()
    