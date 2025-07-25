from terminal import TerminalHandler
from camera import ThreadedCamera
from image import ImageHandler
from ascii import Ascii


from functools import partial
import imageio
import time
import cv2
import os


def cv():
    image = ImageHandler("images/image1.jpg")
    ascii = Ascii("chars/font4x6.png")
    image.fit_chunk(ascii.chunk_dims)

    # testing
    timer = time.perf_counter_ns()
    image.apply(ascii.ascii_print)
    print("time taken:", (time.perf_counter_ns() - timer) * 10**-6, "ms")

    image.show()


def camera():
    cam = ThreadedCamera()
    ascii = Ascii("chars/font6x8.png")

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        image = ImageHandler(frame=frame)
        image.fit_chunk(ascii.chunk_dims)
        image.apply(ascii.ascii_print)
        cv2.imshow("Threaded Camera", image.frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        term = TerminalHandler(frame=frame)
        term.fit()
        term.to_terminal()

    cam.stop()
    cv2.destroyAllWindows()


def gif():
    sequence = [
        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        for frame in imageio.mimread("gifs/gif1.gif")
    ]
    term = TerminalHandler()
    ascii = Ascii("chars/font4x6.png")
    i = 0
    while True:
        image = ImageHandler(frame=sequence[i])
        image.fit_chunk(ascii.chunk_dims)
        image.apply(ascii.ascii_print)
        cv2.imshow("Threaded Camera", image.frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame = sequence[i]
        term.frame = frame
        term.fit()
        term.to_terminal()
        i = (i + 1) % len(sequence)


def main():
    cv()
    gif()
    camera()


if __name__ == "__main__":
    os.system("clear")
    main()
