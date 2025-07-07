# import opencv
import cv2
 
# Read the image
image = cv2.imread('image.jpg')
# grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
cv2.imshow('Original Image', image)
 
cv2.imshow('Grayscale Image', gray_image)
cv2.waitKey(0)
 
cv2.destroyAllWindows()