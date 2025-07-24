import os
import string
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

from frame import Framer

class Ascii():
    def __init__(self, font_path="/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", font_size=16):
        
        self.font = ImageFont.truetype(font_path, font_size)
        self.font_size = font_size
        draw = ImageDraw.Draw(Image.new("L", (100, 100)))
        bbox = draw.textbbox((0, 0), "G", font=self.font)
        self.font_dims = (bbox[3] - bbox[1], bbox[2] - bbox[0])

        norm = lambda char: (np.mean(self.ascii_matrix(char)), char)
        tuple_list = [norm(char) for char in string.printable]
        self.sorted = np.array([char for  _, char in sorted(tuple_list, key=lambda x: x[0])])
            
    def ascii_matrix(self, char):
        img = Image.new("L", self.font_dims, color=255)
        draw = ImageDraw.Draw(img)
        draw.text((0, 0), char, font=self.font, fill=0)
        return np.array(img)
    
    def show(self, char):
        cv2.imshow("Char", self.ascii_matrix(char))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def __getitem__(self, intensity):
        return self.sorted[int(np.round(((1 - intensity) * (len(self.sorted)-1)),0))]

def main():
    image = Framer()
    ascii = Ascii(font_size=23)
    
    image.update_frame(cv2.imread('images/image1.jpg'))
    image.downscale(ascii.font_dims)
    print(ascii.font_dims)

    timer = time.perf_counter_ns()
    image.to_terminal(ascii)
    print("time taken:", (time.perf_counter_ns() - timer) * 10**-6, "ms")

    
    
    #print(np.shape(image.frame[0][0]))
    
    #print(np.shape(image.frame))
    #image.upscale()
    #image.show()

if __name__ == "__main__":
    os.system('clear')
    main()