import imageio
import numpy as np
import cv2
from image import ImageHandler
from ascii import Ascii
import os


class GifHandler(ImageHandler):
    def __init__(self, path):
        sequence = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
                    for frame in imageio.mimread(path)]
        self.sequence = sequence
        self.shape = (len(self.sequence),) + np.shape(self.sequence[0])
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
    
    def show(self):
        frames = 30
        i = 0
        while True:
            frame = self.sequence[i % len(self.sequence)]
            cv2.imshow("GIF", frame)
            if cv2.waitKey(1000 // frames) & 0xFF == ord('q'):
                break

            i += 1

        cv2.destroyAllWindows()

def main():
    os.system('clear')

    ascii = Ascii('chars/font12x16.png')
    height, width = ascii.chunk_dims    
    gif = GifHandler('gifs/gif1.gif')
    gif.fit_chunk(height, width)
    print(gif.shape)
    gif.show()

main()

