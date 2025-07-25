import cv2
import threading
import time
import os
from functools import partial
import imageio

from image import ImageHandler
from ascii import Ascii
from terminal import TerminalHandler

# TODO kolla igenom vad allt detta innebär och applicera på ditt program

class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                if ret:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy()

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()