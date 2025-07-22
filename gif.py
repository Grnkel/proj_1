import imageio
import cv2

gif = imageio.mimread('gifs/gif1.gif')  # Reads all frames

frames = 60
i = 0
while True:
    frame = gif[i % len(gif)]
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow('GIF Frame', frame_bgr)

    # Wait for 100ms between frames (adjust as needed)
    if cv2.waitKey(1000//frames) & 0xFF == ord('q'):
        break

    i += 1

cv2.destroyAllWindows()

