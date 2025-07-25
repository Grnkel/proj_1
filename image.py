from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
import numpy as np
import cv2


class ImageHandler:
    def __init__(self, path=None, frame=None):
        if frame is not None:
            self.frame = frame
        elif path is not None:
            self.frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            self.frame = None


    def __iter__(self):
        for i in range(self.slices[0]):
            for j in range(self.slices[1]):
                yield (i, j), self.__getitem__((i, j))

    def _index(self, index):
        row, col = index
        rows = slice(row * self.chunk_dims[0], (row + 1) * self.chunk_dims[0])
        cols = slice(col * self.chunk_dims[1], (col + 1) * self.chunk_dims[1])
        return rows, cols

    def __getitem__(self, index):
        rows, cols = self._index(index)
        return self.frame[rows, cols]

    def __setitem__(self, index, value):
        rows, cols = self._index(index)
        self.frame[rows, cols] = value

    def show(self):
        cv2.imshow("", self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def grayscale(self):
        self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        return self

    def fit_chunk(self, chunk_dims=(1, 1)):
        ch_height, ch_width = chunk_dims

        img_height, img_width = self.frame.shape[:2]
        v_slices = -(-img_height // ch_height)
        h_slices = -(-img_width // ch_width)

        new_height = v_slices * ch_height
        new_width = h_slices * ch_width
        self.frame = cv2.resize(
            self.frame, (new_width, new_height), interpolation=cv2.INTER_AREA
        )

        self.chunk_dims = (ch_height, ch_width)
        self.slices = (v_slices, h_slices)
        self.chunks = v_slices * h_slices
        self.chunk_pixels = ch_height * ch_width

        return self

    def apply(self, func):
        for index, chunk in self.__iter__():
            self.__setitem__(index, func(chunk))

    def contrast(self, k, hw, chunk):
        avg = np.mean(chunk, axis=(0, 1)) / 255
        exp = np.exp(k * (avg - hw))
        sigmoid = exp / (1 + exp)
        new_values = sigmoid * 255
        return new_values
