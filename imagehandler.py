from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
import numpy as np
import cv2

# TODO klass för pararell bildanalys

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
        self.chunk_dims = None, None
        self.slices = None, None

    def show(self):
        cv2.imshow('', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def grayscale(self):
        self.update(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))

    def extend(self, dx=0, dy=0):
        im = self.image
        if dx > 0:
            for x in range(dx):
                i = 0 if x % 2 == 0 else -1
                im = np.insert(im,i,im[:,i],axis=1)
        else:
            start = int(np.floor(-dx/2))
            end = self.dims[1] - int(np.ceil(-dx/2))
            im = im[:][start:end]
            print(self.dims)
            print(start,end)

        if dy > 0:
            for y in range(dy):
                i = 0 if y % 2 == 0 else -1
                im = np.insert(im,i,im[i,:],axis=0)
        else:
            start = int(np.floor(-dy/2))
            end = self.dims[0] - int(np.ceil(-dy/2))
            im = im[start:end][:]
        self.update(im)
    
def testing():
    image = ImageHandler('images/image1.jpg')
    image.grayscale()
    image.extend(-100)
    
    image.show()

testing()
    