import sys
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def line(p1, p2):                           # function takes two points and calculates y = ax + b equation
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0] * p2[1] - p2[0] * p1[1])
    return A, B, -C


def intersection(L1, L2):                   # function takes two lines and calculates intersection points
    D = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False


def calculateArea(p1, p2):                  # function calculates rectangle area from given two corner points
    height = p1[0] - p2[0]
    width = p1[1] - p2[1]
    return round(abs(height * width))


def myHoughTransformMethod():

    # initializing phase
    height, width = edges.shape
    diagonal_length = int(np.ceil(np.sqrt(width * width + height * height)))
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    rhos = np.arange(-diagonal_length, diagonal_length)
    accumulator = np.zeros((2 * diagonal_length, len(thetas)))

    # voting accumulator phase
    for i in range(width):
        for j in range(height):
            if edges[j][i] == 255:
                for k in range(len(thetas)):
                    rho = round(i * np.cos(thetas[k]) + j * np.sin(thetas[k]))
                    accumulator[rho + diagonal_length][k] += 1

    horizontal_lines = []
    vertical_lines = []

    # filtering horizontal and vertical potential lines
    for i in range(len(accumulator)):
        for k in range(len(accumulator[0])):
            if 50 > accumulator[i][k] > 30 and 92 > k > 88:
                rho = rhos[i]
                theta = thetas[k]
                if len(vertical_lines) == 0 or rho - 50 > vertical_lines[-1][0]:
                    vertical_lines.append([rho, theta])

            if 50 > accumulator[i][k] > 30 and (2 > k or k > 178):
                rho = rhos[i]
                theta = thetas[k]
                if len(horizontal_lines) == 0 or rho - 50 > horizontal_lines[-1][0]:
                    horizontal_lines.append([rho, theta])

    # according to filtered lines I took just 4 rho and theta values for drawing 4 lines
    rho1 = vertical_lines[len(vertical_lines) // 2 - 1][0]
    rho2 = vertical_lines[len(vertical_lines) // 2 + 1][0]
    rho3 = horizontal_lines[int(np.floor(len(horizontal_lines) // 4 * 1))][0]
    rho4 = horizontal_lines[int(np.floor(len(horizontal_lines) // 4 * 1 - 1))][0]
    theta1 = vertical_lines[len(vertical_lines) // 2 - 1][1]
    theta2 = vertical_lines[len(vertical_lines) // 2 + 1][1]
    theta3 = horizontal_lines[int(np.floor(len(horizontal_lines) // 4 * 1))][1]
    theta4 = horizontal_lines[int(np.floor(len(horizontal_lines) // 4 * 1 - 1))][1]

    lines = [rho1, theta1, rho2, theta2, rho3, theta3, rho4, theta4]
    rectangleLines = []

    # according to 4 rho and theta values I transform them to cartesian coordinates
    for i in range(0, len(lines), 2):
        rho = lines[i]
        theta = lines[i + 1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)
        rectangleLines.append([x1, y1, x2, y2])

    # According to 4 line I calculate 2 intersection points and draw rectangle
    L1 = line([rectangleLines[0][0], rectangleLines[0][1]], [rectangleLines[0][2], rectangleLines[0][3]])
    L2 = line([rectangleLines[2][0], rectangleLines[2][1]], [rectangleLines[2][2], rectangleLines[2][3]])
    L3 = line([rectangleLines[1][0], rectangleLines[1][1]], [rectangleLines[1][2], rectangleLines[1][3]])
    L4 = line([rectangleLines[3][0], rectangleLines[3][1]], [rectangleLines[3][2], rectangleLines[3][3]])
    p1 = intersection(L1, L2)
    p2 = intersection(L3, L4)
    cv2.rectangle(img, (round(p1[0]), round(p1[1])), (round(p2[0]), round(p2[1])), (0, 0, 255), 3)
    readXML_and_calculateIOU(p1, p2)

    return


def readXML_and_calculateIOU(p1, p2):                           # this function reads xml file and calculates IOU score
    tree = ET.parse("annotations/" + sys.argv[1] + ".xml")
    root = tree.getroot()
    real_xmin = int(root[4][5][0].text)
    real_ymin = int(root[4][5][1].text)
    real_xmax = int(root[4][5][2].text)
    real_ymax = int(root[4][5][3].text)
    predicted_xmin = min(round(p1[0]), round(p2[0]))
    predicted_xmax = max(round(p1[0]), round(p2[0]))
    predicted_ymin = min(round(p1[1]), round(p2[1]))
    predicted_ymax = max(round(p1[1]), round(p2[1]))
    overlap_point_x = [max(predicted_xmin, real_xmin), min(predicted_ymax, real_ymax)]
    overlap_point_y = [min(predicted_xmax, real_xmax), max(predicted_ymin, real_ymin)]
    overlapArea = calculateArea(overlap_point_x, overlap_point_y)
    unionArea = calculateArea([real_xmin, real_ymin], [real_xmax, real_ymax]) + calculateArea(p1, p2) - overlapArea
    print("IOU SCORE: ", overlapArea / unionArea)
    print("-------------------------------")
    print("PLOT PRINTED TO 'OUTPUT.JPG'")


img = cv2.imread("images/" + sys.argv[1] + ".png")          # reading image path
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # taking gray version for better edge detection
edges = cv2.Canny(gray, 75, 150)                            # canny edge detection applied
cv2.imwrite("edges.jpg", edges)
myHoughTransformMethod()                                    # my hough transform method applied
cv2.imwrite('output.jpg', img)                              # wrote output to output.jpg
