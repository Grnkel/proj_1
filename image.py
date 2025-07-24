from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
import numpy as np
import cv2


class ImageHandler:
    def __init__(self, path):
        self.image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

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
        return self.image[rows, cols]

    def __setitem__(self, index, value):
        rows, cols = self._index(index)
        self.image[rows, cols] = value

    def show(self):
        cv2.imshow("", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self

    def extend(self, dy=0, dx=0):
        im = self.image

        # vertical extension or cropping (dy)
        if dy > 0:
            pad_top = dy // 2
            pad_bottom = dy - pad_top
            if len(np.shape(self.image)) > 2:
                im = np.pad(im, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode="edge")
            else:
                im = np.pad(im, ((pad_top, pad_bottom), (0, 0)), mode="edge")
        elif dy < 0:
            crop_top = -dy // 2
            crop_bottom = -dy - crop_top
            im = im[crop_top:-crop_bottom, :]

        # horizontal extension or cropping (dx)
        if dx > 0:
            pad_left = dx // 2
            pad_right = dx - pad_left
            if len(np.shape(self.image)) > 2:
                im = np.pad(im, ((0, 0), (pad_left, pad_right), (0, 0)), mode="edge")
            else:
                im = np.pad(im, ((0, 0), (pad_left, pad_right)), mode="edge")
        elif dx < 0:
            crop_left = -dx // 2
            crop_right = -dx - crop_left
            im = im[:, crop_left:-crop_right]

        self.image = im

    def fit_chunk(self, chunk_dims=(1, 1)):
        ch_height, ch_width = chunk_dims
        v_slices = -(-np.shape(self.image)[0] // ch_height)
        h_slices = -(-np.shape(self.image)[1] // ch_width)
        diff_y = ch_height * v_slices - np.shape(self.image)[0]
        diff_x = ch_width * h_slices - np.shape(self.image)[1]
        self.extend(diff_y, diff_x)
        self.chunk_dims = ch_height, ch_width
        self.slices = v_slices, h_slices
        self.chunks = v_slices * h_slices
        self.chunk_pixels = ch_height * ch_width
        return self

    def temp(self):
        h, w = np.shape(self.image)[:2]
        ch, cw = self.chunk_dims
        new_h, new_w = h // ch, w // cw
        self.frame = cv2.resize(
            self.frame,
        )

        downscale = cv2.resize(
            self.image, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )

    def apply(self, func):
        for index, chunk in self.__iter__():
            self.__setitem__(index, func(chunk))

    def contrast(self, k, hw, chunk):
        avg = np.mean(chunk, axis=(0, 1)) / 255
        exp = np.exp(k * (avg - hw))
        sigmoid = exp / (1 + exp)
        new_values = sigmoid * 255
        return new_values
