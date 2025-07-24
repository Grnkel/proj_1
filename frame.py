import cv2
import numpy as np

class Framer():
    def __init__(self):
        self.frame = None

    def show(self, delay=0, window_name="Frame"):
        cv2.imshow(window_name, self.frame)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()

    def downscale(self, chunk_dims=(1, 1)):
        self._original_shape = self.frame.shape[:2]
        h, w = self._original_shape
        ch, cw = chunk_dims
        new_h, new_w = h // ch, w // cw
        self.frame = cv2.resize(
            self.frame, 
            (new_w, new_h), 
            interpolation=cv2.INTER_NEAREST)
        return self
    
    def upscale(self, scale=None):
        if scale is not None:
            h, w = self.frame.shape[:2]
            new_w = int(w * scale[1])
            new_h = int(h * scale[0])
            self.frame = cv2.resize(
                self.frame, 
                (new_w, new_h), 
                interpolation=cv2.INTER_NEAREST)
        elif hasattr(self, '_original_shape'):
            h, w = self._original_shape
            self.frame = cv2.resize(
                self.frame, 
                (w, h), 
                interpolation=cv2.INTER_NEAREST)
        return self

    def update_frame(self, frame):
        self.frame = frame

    def to_terminal(self, ascii):
        matrix = []
        for row in range(self.frame.shape[0]):
            line = []
            for col in range(self.frame.shape[1]):
                char = ascii[np.mean(self.frame[row][col] / 255)]
                if np.shape(self.frame[row][col]) == ():
                    r,g,b = np.ones(3)
                else:
                    r,g,b = self.frame[row][col][:3]
                line.append(f"\033[38;2;{b};{g};{r}m{char}\033[0m")
            matrix.append(''.join(line))
        print('\n'.join(matrix))