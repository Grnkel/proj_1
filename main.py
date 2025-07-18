from imagehandler import ImageHandler
from functools import partial
from ascii import Ascii
import time

def custom(image,ascii,row,col):
    #image[row,col] = image.contrast(10,0.6,row,col)
    image[row,col] = ascii.ascii_print(image,row,col)
    return image[row,col]

def main():
    image = ImageHandler('images/image1.jpg')
    ascii = Ascii('chars/font4x6.png')
    ascii.generate_list()
    #image.grayscale() # TODO, ska ändå vara en fungerande funktion
    height, width = ascii.chunk_dims
    image.fit_chunk(height, width)

    timer = time.perf_counter_ns()
    image.apply(func=partial(custom, image, ascii))
    print("time taken:", (time.perf_counter_ns() - timer) * 10**-6, "ms")
    image.show()

main()



