import numpy as np
import pandas as pd
import cv2
from collections import deque

# declaring global variables (are used later on)
clicked = False
r = 0
g = 0
b = 0
colors = []
lst_color_name = []


# Reading the image with opencv
def read_image(image_path):
    img = cv2.imread(image_path)

    return img


# Reading csv file with pandas and giving names to each column
def read_csv():
    index = ["color", "color_name", "hex", "R", "G", "B"]
    csv = pd.read_csv('colors.csv', names=index, header=None)
    return csv


# function to calculate minimum distance from all colors and get the most matching color
def get_color_name(R, G, B, csv):
    minimum = 10000
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if d <= minimum:
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname


# function to get x,y coordinates of mouse double click
def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, clicked, colors
        clicked = True
        img = param
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)
        colors.append((b, g, r))
        print(r, g, b)


def get_colors():
    global clicked, r, b, g, lst_color_name

    # Read image
    img = read_image("Images/mountains.jpg")

    # Read CSV file
    csv = read_csv()

    # Create a window for image
    image_window_name = "image"
    create_window(image_window_name)
    cv2.setMouseCallback(image_window_name, draw_function, param=img)

    while 1:

        cv2.imshow("image", img)
        if clicked:

            # cv2.rectangle(image, startpoint, endpoint, color, thickness)-1 fills entire rectangle
            cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)

            # Creating text string to display( Color name and RGB values )
            color_name = get_color_name(r, g, b, csv)
            text = color_name + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
            lst_color_name.append(color_name)

            # cv2.putText(img,text,start,font(0-7),fontScale,color,thickness,lineType )
            cv2.putText(img, text, (30, 50), 2, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # For very light colours we will display text in black colour
            if r + g + b >= 600:
                cv2.putText(img, text, (30, 50), 2, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            clicked = False

        # Break the loop when four colors are selected
        if cv2.waitKey(20) & len(lst_color_name) == 4:
            break

    cv2.destroyAllWindows()


# default called trackbar function
def set_values(x):
    print("")


# Creating a window using opencv
def create_window(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)


# FUnction to create trackbar
def create_trackbar(window_name, u_hue, u_sat, u_val, l_hue, l_sat, l_val):
    cv2.createTrackbar("Upper Hue", window_name,
                       u_hue, 255, set_values)
    cv2.createTrackbar("Upper Saturation", window_name,
                       u_sat, 255, set_values)
    cv2.createTrackbar("Upper Value", window_name,
                       u_val, 255, set_values)
    cv2.createTrackbar("Lower Hue", window_name,
                       l_hue, 255, set_values)
    cv2.createTrackbar("Lower Saturation", window_name,
                       l_sat, 255, set_values)
    cv2.createTrackbar("Lower Value", window_name,
                       l_val, 255, set_values)


# Get tracker position
# Getting the updated positions of the trackbar and setting the HSV values
def get_trackbar_position(window_name):
    u_hue = cv2.getTrackbarPos("Upper Hue",
                               window_name)
    u_saturation = cv2.getTrackbarPos("Upper Saturation",
                                      window_name)
    u_value = cv2.getTrackbarPos("Upper Value",
                                 window_name)
    l_hue = cv2.getTrackbarPos("Lower Hue",
                               window_name)
    l_saturation = cv2.getTrackbarPos("Lower Saturation",
                                      window_name)
    l_value = cv2.getTrackbarPos("Lower Value",
                                 window_name)

    upper_hsv = np.array([u_hue, u_saturation, u_value])
    lower_hsv = np.array([l_hue, l_saturation, l_value])
    return upper_hsv, lower_hsv


# Function to build color detector for marker
def color_detector(color, window_name):
    u_hue, u_sat, u_val, l_hue, l_sat, l_val = 0, 0, 0, 0, 0, 0
    if color.lower() == "red":
        u_hue, u_sat, u_val = 10, 255, 255
        l_hue, l_sat, l_val = 0, 120, 70
    elif color.lower() == "blue":
        u_hue, u_sat, u_val = 153, 255, 255
        l_hue, l_sat, l_val = 64, 120, 176

    # Create a new window for color detector
    create_window(window_name=window_name)

    # Create trackbar for adjusting the color
    create_trackbar(window_name=window_name,
                    u_hue=u_hue, u_sat=u_sat, u_val=u_val,
                    l_hue=l_hue, l_sat=l_sat, l_val=l_val)


# Create a rectangle on the frame
# Adding the colour buttons to the frame for colour access
def create_rectangle_add_text(frame_name, colors, color_names):
    frame_name = cv2.rectangle(frame_name, (40, 1), (140, 65),
                               (122, 122, 122), -1)
    frame_name = cv2.rectangle(frame_name, (160, 1), (255, 65),
                               colors[0], -1)
    frame_name = cv2.rectangle(frame_name, (275, 1), (370, 65),
                               colors[1], -1)
    frame_name = cv2.rectangle(frame_name, (390, 1), (485, 65),
                               colors[2], -1)
    frame_name = cv2.rectangle(frame_name, (505, 1), (600, 65),
                               colors[3], -1)

    cv2.putText(frame_name, "CLEAR ALL", (49, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(frame_name, color_names[0], (185, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame_name, color_names[1], (298, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame_name, color_names[2], (420, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (0, 0, 0), 1, cv2.LINE_AA)

    cv2.putText(frame_name, color_names[3], (520, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (0, 0, 0), 1, cv2.LINE_AA)

    return frame_name


def main():
    global lst_color_name, colors
    # Getting the colors from uploaded image
    get_colors()

    # Creating the trackbars needed for adjusting the marker colour.
    color_detector_win_name = "Color Detector"
    color_detector("blue", window_name=color_detector_win_name)

    # Giving different arrays to handle colour points of different colour.
    # These arrays will hold the points of a particular colour in the array which will further be used to draw on canvas
    color1_points = [deque(maxlen=1024)]
    color2_points = [deque(maxlen=1024)]
    color3_points = [deque(maxlen=1024)]
    color4_points = [deque(maxlen=1024)]

    # These indexes will be used to mark position of pointers in colour array
    c1_index = 0
    c2_index = 0
    c3_index = 0
    c4_index = 0

    # The kernel to be used for dilation purpose
    kernel = np.ones((5, 5), np.uint8)

    # The colours which will be used as ink for the drawing purpose
    color_index = 0

    # Creating a paint window
    paint_win_name = "Paint Window"
    create_window(paint_win_name)

    # Loading the default webcam of PC.
    cap = cv2.VideoCapture(0)

    # Keep looping
    while True:

        # Reading the frame from the camera
        ret, frame = cap.read()

        # Flipping the frame to see same side of yours
        frame = cv2.flip(frame, 1)

        # Here is code for Canvas setup
        paint_window = np.zeros_like(frame) + 255

        # Convert the BGR colors to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Getting the updated positions of the trackbar and setting the HSV values
        upper_hsv, lower_hsv = get_trackbar_position(window_name=color_detector_win_name)

        # Adding the colour buttons to the live frame for colour access
        # Label the rectangular boxes drawn on the image
        frame = create_rectangle_add_text(frame, colors, lst_color_name)

        # Draw buttons like colored rectangles on the white image
        # Label the rectangular boxes drawn on the image
        paint_window = create_rectangle_add_text(paint_window, colors, lst_color_name)

        # Identifying the pointer by making its mask
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Find contours for the pointer after identifying it
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        # If the contours are formed
        if len(cnts) > 0:

            # sorting the contours to find biggest
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

            # Get the radius of the enclosing circle
            # around the found contour
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)

            # Draw the circle around the contour
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

            # Calculating the center of the detected contour
            m = cv2.moments(cnt)
            center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))

            # Now checking if the user wants to click on any button above the screen
            if center[1] <= 65:

                # Clear Button
                if 40 <= center[0] <= 140:
                    color1_points = [deque(maxlen=512)]
                    color2_points = [deque(maxlen=512)]
                    color3_points = [deque(maxlen=512)]
                    color4_points = [deque(maxlen=512)]

                    c1_index = 0
                    c2_index = 0
                    c3_index = 0
                    c4_index = 0

                    paint_window[67:, :, :] = 255
                elif 160 <= center[0] <= 255:
                    color_index = 0  # color 1
                elif 275 <= center[0] <= 370:
                    color_index = 1  # color 2
                elif 390 <= center[0] <= 485:
                    color_index = 2  # color 3
                elif 505 <= center[0] <= 600:
                    color_index = 3  # color 4
            else:
                if color_index == 0:
                    color1_points[c1_index].appendleft(center)
                elif color_index == 1:
                    color2_points[c2_index].appendleft(center)
                elif color_index == 2:
                    color3_points[c3_index].appendleft(center)
                elif color_index == 3:
                    color4_points[c4_index].appendleft(center)

        # Append the next deques when nothing is detected to avoid messing up
        else:
            color1_points.append(deque(maxlen=512))
            c1_index += 1
            color2_points.append(deque(maxlen=512))
            c2_index += 1
            color3_points.append(deque(maxlen=512))
            c3_index += 1
            color4_points.append(deque(maxlen=512))
            c4_index += 1

        # Draw lines of all the colors on the canvas and frame
        points = [color1_points, color2_points, color3_points, color4_points]
        # print(points)
        for i in range(len(points)):

            for j in range(len(points[i])):

                for k in range(1, len(points[i][j])):

                    if points[i][j][k - 1] is None or points[i][j][k] is None:
                        continue

                    cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                    cv2.line(paint_window, points[i][j][k - 1], points[i][j][k], colors[i], 2)

        # Show all the windows
        cv2.imshow("Tracking", frame)
        cv2.imshow(paint_win_name, paint_window)
        cv2.imshow("mask", mask)

        # If the 'q' key is pressed then stop the application
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
