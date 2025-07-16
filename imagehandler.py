from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
import numpy as np
import cv2
import time
import re

# own stuff
from ascii import ascii_dict, ascii_matrix  

class ImageHandler():
    def __init__(self, path):
        self.path = path
        self.image = cv2.imread(path)
        self.dims = np.shape(self.image)
        self.chunk_dims = None, None
        self.slices = None, None

    def __iter__(self):
        height, width = self.dims
        for i in range(0, height, self.chunk_dims[0]):
            for j in range(0, width, self.chunk_dims[1]):
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
        ch_width = -(-self.dims[1] // h_slices)

        diff_y = -self.dims[0] % ch_height
        diff_x = -self.dims[1] % ch_width

        self.extend(diff_y, diff_x)

        self.chunk_dims = ch_height, ch_width
        self.slices = v_slices, h_slices

        #print("diffs:", diff_y, diff_x)
        #print("chunk dims:", ch_height, ch_width)

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

        #print("diffs:", diff_y, diff_x)
        #print("slices:", "v:", v_slices, "h:", h_slices)
    
    def apply(self, cores=cpu_count(), func=None):
        shm = shared_memory.SharedMemory(create=True, size=self.image.nbytes)
        shared_mem_image = np.ndarray(self.image.shape, dtype=self.image.dtype, buffer=shm.buf)
        shared_mem_image[:] = self.image[:]

        partial_func = partial(
            self.parallel,
            shm.name,
            cores,
            func,

        )

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

    def parallel(self, shm_name, cores, cunk_func, task):
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
                #res = np.sum(image[ROW, COL])/(self.chunk_dims[0]*self.chunk_dims[1])
                image[ROW, COL] = cunk_func(row,col)

        shm.close()
    
    def contrast(self, k=1.0, hw=0.5, row=None, col=None):
        if row is None or col is None:
            raise ValueError("row and col slices must be provided")
        h_ch, w_ch = self.chunk_dims
        x = np.sum(self.image[row, col]) / (h_ch * w_ch) / 255
        exp = np.exp(k * (x - hw))
        sigmoid = exp / (1 + exp)
        self.image[row, col] = sigmoid * 255
        return self.image
    
    def checkers(self):
        for i in range(self.slices[0]):
            for j in range(self.slices[1]):
                self[i, j] = 0 if (i % 2 == 0) and (j % 2 == 0) else 255
        return self.image

def ascii_print(image, row, col):
    ascii = ImageHandler('chars/font8x12.png')
    ascii.grayscale()
    ascii.fit_chunk(12,8)
    i,j = ascii_dict["X"]
    normalized = image[row, col] / 255.0 
    return ascii[i,j] * normalized    
    
def testing():
    image = ImageHandler('images/image1.jpg')
    image.grayscale()
    image.fit_chunk(12,8)

    timer = time.perf_counter_ns()

    image.apply(func=partial(ascii_print, image))
    print("time taken:", (time.perf_counter_ns() - timer) * 10**-6, "ms")

    image.show()

class Ascii(ImageHandler): # TODO gör att man kan indexera med "bokstav"
    def __init__(self, path, ascii_matrix):   
        ascii_dict = {
            char: (y, x)
            for y, row in enumerate(ascii_matrix)
            for x, char in enumerate(row)
        }
        self.dict = ascii_dict
        self.matrix = ascii_matrix

        super().__init__(path)
        super().grayscale()
        match = re.match(r"chars/font(\d+)x(\d+).png", path)
        if match:
            width, height = match.groups()
            super().fit_chunk(int(height), int(width))
        else:
            ValueError("Invalid file")

    def __getitem__(self, key):
        return super().__getitem__(self.dict[key])
    
    def __setitem__(self, key, value):
        super().__setitem__(self.dict[key], value)

    def __iter__(self):
        return super().__iter__()

ascii = Ascii('chars/font12x16.png', ascii_dict)
ascii["X"] = 0
ascii_flat = np.squeeze(ascii.matrix)
print(ascii_flat)
print(ascii["X"])

for x in ascii:
    if x == ascii["X"]:
        print(x)
#print(ascii_dict["A"])






    