import numpy as np
import cv2
import re

from image import ImageHandler

ascii_matrix = [
    [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/'],
    ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?'],
    ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O'],
    ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_'],
    ['`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o'],
    ['p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', '^']
]

ascii_dict = {
    char: (y, x)
    for y, row in enumerate(ascii_matrix)
    for x, char in enumerate(row)
}

class Ascii(ImageHandler):
    def __init__(self, path):   
        self.dict = ascii_dict
        self.image = cv2.imread(path)
        self.dims = np.shape(self.image)
        
        match = re.match(r"chars/font(\d+)x(\d+).png", path)
        if match:
            width, height = match.groups()
            super().fit_chunk(int(height), int(width))
        else:
            ValueError("Invalid file")

    def __getitem__(self, key):
        return super().__getitem__(self.dict[key])
    
    def __setitem__(self, key, value):
        super().__setitem__(self.dict[key], value)

    def generate_list(self):
        temp = []
        for key in self.dict:
            sum = np.sum(self.__getitem__(key)) / 256
            temp.append((key,sum))
        self.sorted = [key for key, _ in sorted(temp, key=lambda x: x[1])]

    def ascii_print(self, image, row, col):
        rows = slice(row * self.chunk_dims[0], (row + 1) * self.chunk_dims[0])
        cols = slice(col * self.chunk_dims[1], (col + 1) * self.chunk_dims[1])
        im = image[rows,cols]
        index = np.mean(im) / 256
        index = self.sorted[int(index*len(self.sorted))]
        avg_color = np.mean(im,axis=(0,1)) / 256       
        return self.__getitem__(index) * avg_color

    