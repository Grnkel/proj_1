import cv2
import threading
import time

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

    def pixelate(self, img, pixel_size=16):
        height, width = img.shape[:2]

        # Calculate size of the downscaled image
        w, h = width // pixel_size, height // pixel_size

        # Resize small and then scale back up
        temp = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)


# Usage example
def main():
    cam = ThreadedCamera()

    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cam.pixelate(frame, 16)

        # Process the frame here (e.g., pixelate, detect faces, etc)
        cv2.imshow('Threaded Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

