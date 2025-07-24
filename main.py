from image import ImageHandler
from terminal import TerminalHandler
from gif import GifHandler

from functools import partial
from ascii import Ascii
import time
import numpy as np
import os


def cv2():
    image = ImageHandler("images/image1.jpg")
    ascii = Ascii("chars/font4x6.png")
    image.fit_chunk(ascii.chunk_dims)

    # testing
    timer = time.perf_counter_ns()
    image.apply(ascii.ascii_print)
    print("time taken:", (time.perf_counter_ns() - timer) * 10**-6, "ms")

    image.show()


def term():
    import time
    from functools import partial

    term = TerminalHandler("images/image1.jpg")

    # testing
    timer = time.perf_counter_ns()
    term.apply(partial(term.contrast, 11, 0.5))
    term.to_terminal()
    print("time taken:", (time.perf_counter_ns() - timer) * 10**-6, "ms")


def gif():
    ascii = Ascii("chars/font4x6.png")

    gif = GifHandler("gifs/gif1.gif")
    gif.fit_chunk(ascii.chunk_dims)

    print(gif[0,0,0])

    # gif.sequence = np.array(gif.sequence)[range(0,len(gif.sequence),20)]
    gif.sequence = np.array(gif.sequence)[range(0, len(gif.sequence), 10)]
    gif.apply(ascii.ascii_print)
    gif.show()


def main():
    term()
    cv2()
    gif()


if __name__ == "__main__":
    os.system("clear")
    main()
