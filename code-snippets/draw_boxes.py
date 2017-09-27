import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)

    for box in bboxes:
        cv2.rectangle(draw_img, box[0], box[1], color, thick)
   
    return draw_img 
    
    
# Add bounding boxes in this format, these are just example coordinates.
bboxes = [((900, 700), (1200, 500)), ((300, 500), (400, 600))]

result = draw_boxes(image, bboxes)
plt.imshow(result)



