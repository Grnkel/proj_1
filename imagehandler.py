from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
import numpy as np
import cv2
import time

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
        rows = slice(i * self.chunk_dims[0], (i + 1) * self.chunk_dims[0]+1)
        cols = slice(j * self.chunk_dims[1], (j + 1) * self.chunk_dims[1]+1)
        return self.image[rows, cols]
    
    def __setitem__(self, index, value):
        i, j = index
        rows = slice(i * self.chunk_dims[0], (i + 1) * self.chunk_dims[0]+1)
        cols = slice(j * self.chunk_dims[1], (j + 1) * self.chunk_dims[1]+1)
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
        shared_mem_image[:] = self.image[:]

        partial_func = partial(
            self.parallel,
            shm.name,
            cores
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

    def parallel(self, shm_name, cores, task):
        i, j = task
        shm = shared_memory.SharedMemory(name=shm_name)
        image = np.ndarray(self.image.shape, dtype=self.image.dtype, buffer=shm.buf)

        v_start = int(np.ceil(i))
        v_end = int(np.ceil((i + 1)))
        h_start = int(np.ceil(j * self.slices[1] / cores))
        h_end = int(np.ceil((j + 1) * self.slices[1] / cores))

        for row in range(v_start, v_end):
            for col in range(h_start, h_end):
                ROW = slice(row * self.chunk_dims[0], (row + 1) * self.chunk_dims[0]+1)
                COL = slice(col * self.chunk_dims[1], (col + 1) * self.chunk_dims[1]+1)
                res = np.sum(image[ROW, COL])/(self.chunk_dims[0]*self.chunk_dims[1])
                image[ROW, COL] = res

        shm.close()

# TODO gör parallelliseringen mer generell, 
# att man kan lägga in godtycklig funktion som opererar på chunks

# TODO använd detta för att extrahera ASCII bildens chunks till 
# en lista som kan loopas över för att kolla grejer eller manipulera / 
# kopiera skit

# TODO skapa en iter-instance för denna klass som ger referens och 
# tillgång till varje chunk, likt ovan

# TODO egen klass för ascii som implementerar imageHandler?
    
def testing():
    image = ImageHandler('images/image1.jpg')
    image.grayscale()
    image.fit_chunk(20,20)

    #timer = time.perf_counter_ns()
    #image.apply()
    #print("time taken:", (time.perf_counter_ns() - timer) * 10**-6, "ms")
    for i in range(image.slices[1])[1:-1]:
        image[i,0] = 0 if i % 2 == 0 else 255
    print(image.dims)
    print(image.slices)
    image.show()

testing()
    