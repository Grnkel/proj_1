from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
import numpy as np
import cv2

class ImageHandler():
    def __init__(self, path):
        self.image = cv2.imread(path)
        self.dims = np.shape(self.image)

    def __iter__(self):
        height, width, _ = self.dims
        for i in range(0, height, self.chunk_dims[0]):
            for j in range(0, width, self.chunk_dims[1]):
                print(self.image[i:i+self.chunk_dims[0], j:j+self.chunk_dims[1]])
                yield self.image[i:i+self.chunk_dims[0], j:j+self.chunk_dims[1]]

    def __getitem__(self, index):
        i, j = index
        rows = slice(i * self.chunk_dims[0], (i + 1) * self.chunk_dims[0])
        cols = slice(j * self.chunk_dims[1], (j + 1) * self.chunk_dims[1])
        return self.image[rows, cols]
    
    def __setitem__(self, index, value):
        i, j = index
        rows = slice(i * self.chunk_dims[0], (i + 1) * self.chunk_dims[0])
        cols = slice(j * self.chunk_dims[1], (j + 1) * self.chunk_dims[1])
        self.image[rows, cols] = value

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
            im = np.pad(im, ((pad_top, pad_bottom), (0, 0), (0,0)), mode='edge')
        elif dy < 0:
            crop_top = -dy // 2
            crop_bottom = -dy - crop_top
            im = im[crop_top:- crop_bottom, :]

        # horizontal extension or cropping (dx)
        if dx > 0:
            pad_left = dx // 2
            pad_right = dx - pad_left
            im = np.pad(im, ((0, 0), (pad_left, pad_right), (0,0)), mode='edge')
        elif dx < 0:
            crop_left = -dx // 2
            crop_right = -dx - crop_left
            im = im[:, crop_left:- crop_right]

        self.update(im)

    def fit_chunk(self, ch_height=1, ch_width=1):
        assert(ch_height>0)
        assert(ch_width>0)

        # resizing image to fit with slices
        v_slices = -(-self.dims[0] // ch_height)
        h_slices = -(-self.dims[1] // ch_width)

        diff_y = ch_height * v_slices - self.dims[0]
        diff_x = ch_width * h_slices - self.dims[1]

        self.extend(diff_y, diff_x)

        self.chunk_dims = ch_height, ch_width
        self.slices = v_slices, h_slices
        self.chunks = v_slices * h_slices
        self.chunk_pixels = ch_height * ch_width
    
    def apply(self, cores=cpu_count(), func=None):
        shm = shared_memory.SharedMemory(create=True, size=self.image.nbytes)
        shared_mem_image = np.ndarray(self.image.shape, dtype=self.image.dtype, buffer=shm.buf)
        shared_mem_image[:] = self.image[:]
        partial_func = partial(self.parallel,shm.name,cores,func,)

        space = [
            (i, j)
            for i in range(self.slices[0])
            for j in range(cores)
        ]

        with Pool(processes=cores) as pool:
            pool.map(partial_func, space)

        # read back
        self.image = shared_mem_image.copy()
        shm.close()
        shm.unlink()

    def parallel(self, shm_name, cores, chunk_func, task):
        i, j = task
        shm = shared_memory.SharedMemory(name=shm_name)
        image = np.ndarray(self.image.shape, dtype=self.image.dtype, buffer=shm.buf)

        v_start = int(np.ceil(i))
        v_end = int(np.ceil((i + 1)))
        h_start = int(np.ceil(j * self.slices[1] / cores))
        h_end = int(np.ceil((j + 1) * self.slices[1] / cores))

        for row in range(v_start, v_end):
            for col in range(h_start, h_end):
                ROW = slice(row * self.chunk_dims[0], (row + 1) * self.chunk_dims[0])
                COL = slice(col * self.chunk_dims[1], (col + 1) * self.chunk_dims[1])
                image[ROW, COL] = chunk_func(row,col)

        shm.close()
    
    def contrast(self, k, hw, row, col):
        chunk = self.__getitem__((row, col))
        avg = np.mean(chunk, axis=(0, 1)) / 255
        exp = np.exp(k * (avg - hw))
        sigmoid = exp / (1 + exp)
        new_values = (sigmoid * 255)
        self.image[row, col] = new_values
        return self.image[row, col]

    
# TODO lägg till färg
# TODO gör det till video

# TODO gör den snabbare och mer effektiv (bättre lösning) 
# alltså kanske att ascii skiten är en lista med bools eller 
# något där man faktiskt får veta att det antingen är 0 eller 1 
# ör att verkligen snabba upp beräkningen, kanske först därefter 
# som man kollar färg? testa att sänka precisionen på alla matriser, 
# man behöver ju inte tre kanaler med sådan där stor precision direkt, 
# man kanske kan göra bit-manipulationer eller vem vet vad python kan göra

# TODO gör en gemensam klass som använder siga 





    