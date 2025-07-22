import os
import numpy as np
from ascii import Ascii
from functools import partial
from image import ImageHandler

class TerminalHandler(ImageHandler):
    def __init__(self, path):
        super().__init__(path)

        self.ascii = Ascii("chars/font12x16.png") 
        # TODO kolla hur mycket space (v och h som finns och ta mindre och mindre chars
        self.ascii.generate_list()
        height, width = self.ascii.chunk_dims
        
        width = int(width * 0.60) # ascii to wide in chars/

        size = os.get_terminal_size()
        v_slices = -(-self.dims[0] // height) # att detta görs på två ställen är lite B
        h_slices = -(-self.dims[1] // width)
        y_diff = (size.lines - v_slices) * height
        x_diff = (size.columns - h_slices) * width
        self.extend(np.min((y_diff,0)), x_diff)      

        self.fit_chunk(height, width)    
        v_slices, h_slices = self.slices    
        self.matrix = np.full((v_slices, h_slices) , '', dtype='<U50')         
    
    def to_terminal(self):
        for i,j,chunk in super().__iter__():
            index = np.mean(chunk,axis=(0,1))
            r,g,b =  [int(np.ceil(v)) for v in index] if np.shape(index) != () else (255,255,255)
            index = np.mean(index) / 255
            char = self.ascii.sorted[int(index*len(self.ascii.sorted))]
            self.matrix[i][j] = f"\033[38;2;{b};{g};{r}m{char}\033[0m"

        os.system('clear')
        for row in self.matrix:
            print(''.join(row))
            

def main():
    terminal = TerminalHandler('images/image1.jpg')

    if False:
        terminal.grayscale().apply(func=partial(terminal.contrast,15,0.5))
    else:
        terminal.apply(func=partial(terminal.contrast,10,0.5))

    terminal.show()
    terminal.to_terminal()

    
main()
    


# TODO gör faktiskt en funktion som kan printa ut till terminalen