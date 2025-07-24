import imageio
import numpy as np
import cv2
from image import ImageHandler
from ascii import Ascii
from functools import partial
import os
from multiprocessing import cpu_count
import time

class GifHandler(ImageHandler):
    def __init__(self, path):
        sequence = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
                    for frame in imageio.mimread(path)]
        self.sequence = sequence

    def __iter__(self):
        for frame in range(len(self.sequence)):
            self.image = frame
            for i in range(self.slices[0]):
                for j in range(self.slices[1]):
                    yield (i,j), self.__getitem__((i, j))

    def fit_chunk(self, chunk_dims):
        for i, frame in enumerate(self.sequence):
            self.image = frame
            super().fit_chunk(chunk_dims)
            self.sequence[i] = self.image 
        return self
    
    def grayscale(self):
        for i, frame in enumerate(self.sequence):
            self.sequence[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return super().grayscale()
    
    def apply(self, func):
        for i, frame in enumerate(self.sequence):
            self.image = frame
            super().apply(func)
            self.sequence[i] = self.image
    
    def show(self):
        frames = 30
        i = 0
        while True:
            frame = self.sequence[i % len(self.sequence)]
            cv2.imshow("GIF", frame)
            if cv2.waitKey(int(1000 // frames)) & 0xFF == ord('q'):
                break
            i += 1
        cv2.destroyAllWindows()

def main():
    ascii = Ascii('chars/font4x6.png')
    ascii.generate_list()

    gif = GifHandler('gifs/gif1.gif')    
    gif.fit_chunk(ascii.chunk_dims)
    gif.grayscale()
    
    #gif.sequence = np.array(gif.sequence)[range(0,len(gif.sequence),20)]
    #gif.sequence = np.array(gif.sequence)[range(0,len(gif.sequence),10)]
    #gif.apply(ascii.ascii_print)
    gif.show()

if __name__ == "__main__":
    os.system('clear')
    main()

