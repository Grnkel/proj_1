import imageio
import numpy as np
import cv2
from image import ImageHandler
from ascii import Ascii
from functools import partial
import os
from multiprocessing import cpu_count
from old.terminal import TerminalHandler
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

    def fit_chunk(self, ch_height=1, ch_width=1):
        for i, image in enumerate(self.sequence):
            self.image = image
            self.dims = np.shape(image)
            super().fit_chunk(ch_height, ch_width)
            self.sequence[i] = self.image 
        return self
    
    def apply(self, cores=cpu_count(), funcs=None):
        for i, image in enumerate(self.sequence):
            self.image = image
            self.dims = np.shape(image)
            timer = time.perf_counter_ns()
            super().apply(cores,funcs[i])
            print(i,"\ttime taken:", (time.perf_counter_ns() - timer) * 10**-6, "ms")
            self.sequence[i] = self.image
        return self
    
    def show(self):
        frames = 30 / 5
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
    height, width = ascii.chunk_dims    

    gif = GifHandler('gifs/gif1.gif')    
    gif.fit_chunk(height, width)
    gif.sequence = np.array(gif.sequence)[range(0,len(gif.sequence),5)]
    funcs = [partial(ascii.ascii_print, image) for image in gif.sequence]
    gif.apply(cores=1,funcs=funcs)
    gif.show()

main()

