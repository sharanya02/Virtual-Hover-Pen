from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import cv2

class ColorLabeler:
    def __init__(self):
        colors = OrderedDict({
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
        })

        # Allocate memory for the L*A*B image, then initialize the color names list
        self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
        self.colorNames = []

        # loop over the colors dictionary
        for (i, (name, rgb)) in enumerate(colors.items()):
            self.lab[i] = rgb
            self.colorNames.append(name)

        # Convert L*A*B array from RGB color space to L*A*B
        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

    def label(self, frame, c):
        # Construct a mask for the contour, then compute the
        # Average LAB value for the masked region
        mask = np.zeros(frame.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(frame, mask=mask)[:3]

        # Initialize the minimum distance found thus far
        minDist = (np.inf, None)

        # loop over the known LAB color values
        for (i, row) in enumerate(self.lab):
            # compute the distance between the current LAB color value and the mean of the image
            d = dist.euclidean(row[0], mean)

            # If the distance is smaller than the current distance,
            # then update the bookkeeping value
            if d < minDist[0]:
                minDist = (d, i)

        # Return the name of the color with the smallest distance
        return self.colorNames[minDist[1]]
