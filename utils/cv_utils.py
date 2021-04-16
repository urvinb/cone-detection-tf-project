from __future__ import division

import cv2
import numpy as np

# Draw bounding box with text above the box
def add_rectangle_with_text(image, ymin, xmin, ymax, xmax, color, text):
    """
    Draw a bounding box with text above the box
    Args:
        image : image on which box needs to be drawn
        ymin, xmin, ymax, xmax: co-ordinates of the box
        color: Contanins the colour of the boundiing box to be drawn
        text: Probability that is displayed above the box
    """
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 5)
    cv2.putText(image, text, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2,
                cv2.LINE_AA)
