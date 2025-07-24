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
        self.dims = np.shape(self.sequence[0])

    def __iter__(self):
        for frame in self.sequence:
            self.image = frame
            self.dims = np.shape(frame)
            yield from super().__iter__()

    def fit_chunk(self, chunk_dims):
        for i, image in enumerate(self.sequence):
            self.image = image
            self.dims = np.shape(image)
            super().fit_chunk(chunk_dims)
            self.sequence[i] = self.image 
        return self
    
    def apply(self, func):
        for i, image in enumerate(self.sequence):
            self.image = image
            self.dims = np.shape(image)
            timer = time.perf_counter_ns()
            super().apply(func)
            print(i,"\ttime taken:", (time.perf_counter_ns() - timer) * 10**-6, "ms")
            self.sequence[i] = self.image
        return self
    
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
    os.system('clear')

    ascii = Ascii('chars/font4x6.png')
    ascii.generate_list()

    gif = GifHandler('gifs/gif1.gif')    
    gif.fit_chunk(ascii.chunk_dims)
    
    #gif.sequence = np.array(gif.sequence)[range(0,len(gif.sequence),20)]
    gif.sequence = np.array(gif.sequence)
    np.shape(gif.sequence)
    #gif.apply(ascii.ascii_print)
    gif.show()

main()

