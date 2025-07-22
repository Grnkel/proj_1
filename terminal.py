import os
from image import ImageHandler
import numpy as np
from ascii import Ascii
from functools import partial

# Get terminal size
size = os.get_terminal_size()
print(f"Columns: {size.columns}, Rows: {size.lines}")

text = "Hello\nWorld"
lines = text.splitlines()
height = len(lines)
width = max(len(line) for line in lines)

print(f"Width: {width}, Height: {height}")


class TerminalHandler(ImageHandler):
    def __init__(self, path):
        super().__init__(path)
        self.grayscale()

        self.ascii = Ascii("chars/font12x16.png")
        self.ascii.generate_list()
        
        height, width = self.ascii.chunk_dims
        self.fit_chunk(height, width+2) # sliding window effect
        print(self.chunks)
        print(self.slices)
            
        v_slices, h_slices = self.slices
        self.matrix = np.full((v_slices, h_slices) , '', dtype='<U1') 

        
    
    def to_terminal(self):
        for i,j,chunk in super().__iter__():
            index = np.sum(chunk) / self.chunk_pixels / 256 / 3
            char = self.ascii.sorted[int(index*len(self.ascii.sorted))]
            self.matrix[i][j] = char

        os.system('clear')
        for row in self.matrix:
            print(' '.join(row))

def main():
    terminal = TerminalHandler('images/image4.jpg')
    terminal.apply(func=partial(terminal.contrast,15,0.8))
    terminal.show()
    terminal.to_terminal()

    
main()
    


# TODO gör faktiskt en funktion som kan printa ut till terminalen