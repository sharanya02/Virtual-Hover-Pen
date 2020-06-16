from color_labeler import ColorLabeler
import numpy as np
import cv2
import imutils
import time


def main():
    # define the lower and upper boundaries of the colors in HSV color space
    blueLower = (69, 165, 169)
    blueUpper = (134, 243, 255)

    greenLower = (63, 114, 120)
    greenUpper = (100, 255, 255)

    redLower = (121, 190, 91)
    redUpper = (206, 255, 255)

    r = (0, 0, 255)
    g = (0, 255, 0)
    b = (255, 0, 0)
    x1, y1 = 0, 0

    wiper_thresh = 40000
    clear = False
    thickness = 4
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cl = ColorLabeler()

    # allow the camera or video file to warm up
    time.sleep(2.0)
    canvas = None

    while True:
        # Grab the current frame
        # frame = vs.read()
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if frame is None:
            break

        # Blur the frame
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        # Convert to L*a*b* color space, used in color_labeler.py
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

        # Convert to the HSV color space
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        if canvas is None:
            canvas = np.zeros_like(frame)

        # Construct masks
        redMask = cv2.inRange(hsv, redLower, redUpper)
        blueMask = cv2.inRange(hsv, blueLower, blueUpper)
        greenMask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = greenMask | blueMask | redMask

        # Perform a series of dilations and erosions to remove any small blobs
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours in the mask and initialize the current
        # (x, y) center of the object
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Only proceed if at least one contour is found
        if len(cnts) > 0:
            # Find the largest contour in the mask,
            # then use it to compute the minimum enclosing circle and centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            x2, y2 = center

            if area < 2000:
                thickness = 0
            elif area < 2500:
                thickness = 4
            elif area < 3000:
                thickness = 6
            elif area < 4000:
                thickness = 8
            elif area < 6000:
                thickness = 10
            elif area < 10000:
                thickness = 14
            elif area < 12000:
                thickness = 18
            # Detect current color using color_labeler.py
            detected_color = (cl.label(lab, c))
            if detected_color == "blue":
                draw_color = b
            elif detected_color == "red":
                draw_color = r
            elif detected_color == "green":
                draw_color = g
            else:
                draw_color = None
            print(detected_color)

            if x1 == 0 and y1 == 0:
                x1, y1 = x2, y2
            elif thickness > 1:
                canvas = cv2.line(canvas, (x1, y1), (x2, y2), draw_color, thickness)
            x1, y1, = x2, y2

            # Puts circle at center of marker, the point where the drawing will be done
            # cv2.circle(frame, center, 5, (0, 255, 255), -1)

            # If the marker is close enough to the screen, it clears it
            if area > wiper_thresh:
                cv2.putText(canvas, 'Clearing Canvas', (100, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5, cv2.LINE_AA)
                clear = True
        else:
            # If there were no contours detected, make x1, y1 = 0
            x1, y1 = 0, 0

        # Put canvas on top of the frame
        _, mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY)
        foreground = cv2.bitwise_and(canvas, canvas, mask=mask)
        background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
        frame = cv2.add(foreground, background)

        # Show the frame to the sreen
        cv2.imshow("Virtual Pen", frame)

        # If q is pressed, video stops
        close_key = cv2.waitKey(1) & 0xFF
        if close_key == ord("q"):
            break
        if clear:
            time.sleep(1)
            canvas = None
            clear = False

    # When loop ends, exit program
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
