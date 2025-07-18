import os
from imagehandler import ImageHandler
import numpy as np
from ascii import Ascii

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
        # TODO fixa skiten så den är automatisk!
        self.ascii = Ascii("chars/font12x16.png")
        self.ascii.generate_list()

        self.matrix = np.arange()
        for chunk in super().__iter__():
            print(chunk)

def main():
    image = ImageHandler('images/image4.jpg')
    ascii = Ascii('chars/font4x6.png')
    
main()
    


# TODO gör faktiskt en funktion som kan printa ut till terminalen