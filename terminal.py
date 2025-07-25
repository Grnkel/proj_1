import os
import numpy as np
from ascii import Ascii
from image import ImageHandler


class TerminalHandler(ImageHandler):
    def __init__(self, path=None, frame=None):
        super().__init__(path, frame)
        self.fitted = False

    def fit(self):
        if self.fitted:
            return self

        size = os.get_terminal_size()

        min_diffs = (np.inf, np.inf, "")
        for filename in os.listdir("chars"):

            self.ascii = Ascii("chars/" + filename)
            height, width = self.ascii.chunk_dims
            y_diff = (size.lines + (-np.shape(self.frame)[0] // height)) * height
            x_diff = (size.columns + (-np.shape(self.frame)[1] // width)) * width

            if x_diff >= 0 and y_diff >= 0 and x_diff < min_diffs[1]:
                min_diffs = (y_diff, x_diff, "chars/" + filename)

        y_diff, x_diff, size = min_diffs
        self.ascii = Ascii(size)

        self.fit_chunk(self.ascii.chunk_dims)
        self.matrix = np.full(self.slices, "", dtype="<U50")
        self.fitted = True
        return self

    def to_terminal(self):
        for (i, j), chunk in super().__iter__():
            index = np.mean(chunk, axis=(0, 1))
            r, g, b = (
                [int(np.ceil(v)) for v in index]
                if np.shape(index) != ()
                else (255, 255, 255)
            )
            index = np.mean(index) / 255
            char = self.ascii.sorted[int(index * (len(self.ascii.sorted) - 1))]
            self.matrix[i][j] = f"\033[38;2;{b};{g};{r}m{char}\033[0m"
        os.system("clear")
        for row in self.matrix:
            print("".join(row))
        return self
