import numpy as np
import cv2



def computeGrad(img,threshold = 80):
    '''
    Calcul les gradients de l'image
    :param im: input BGR image
    :return:
        magnitude of gradient, orientation of gradient (with mask)
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate gradient magnitude and orientation
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_ori = np.arctan2(grad_y, grad_x) * 180 / np.pi


    # Apply threshold to gradient magnitude

    _, mask = cv2.threshold(grad_mag, threshold, 255, cv2.THRESH_BINARY)

    # Apply mask to orientations
    grad_ori_masked = cv2.bitwise_and(grad_ori, grad_ori, mask=mask.astype(np.uint8))

    return grad_ori_masked,grad_mag
def buildRefTable(img):
    """
    FROM : https://github.com/adl1995/generalised-hough-transform

    builds the reference table for the given input template image
    :param im: input binary image
    :return:
        table = a reconstructed reference table...
    """
    table = [[0 for x in range(1)] for y in range(90)]  # creating a empty list
    # r will be calculated corresponding to this point
    img_center = [int(img.shape[0]/2), int(img.shape[1]/2)]

    def findAngleDistance(x1, y1):
        x2, y2 = img_center[0], img_center[1]
        r = [(x2-x1), (y2-y1)]
        if (x2-x1 != 0):
            return [int(np.rad2deg(np.arctan(int((y2-y1)/(x2-x1))))), r]
        else:
            return [0, 0]

    filter_size = 3
    for x in range(img.shape[0]-(filter_size-1)):
        for y in range(img.shape[1]-(filter_size-1)):
            if (img[x, y] != 0.):
                theta, r = findAngleDistance(x, y)
                if (r != 0):
                    table[np.absolute(theta)].append(r)

    for i in range(len(table)):
        table[i].pop(0)

    return table

def matchTable(im, table):
    """
    FROM : https://github.com/adl1995/generalised-hough-transform

    :param im: input binary image, for searching template
    :param table: table for template
    :return:
        accumulator with searched votes
    """
    # matches the reference table with the given input
    # image for testing generalized Hough Transform
    m, n = im.shape
    acc = np.zeros((m+50, n+50))  # acc array requires some extra space

    def findGradient(x, y):
        if (x != 0):
            return int(np.rad2deg(np.arctan(int(y/x))))
        else:
            return 0

    for x in range(1, im.shape[0]):
        for y in range(im.shape[1]):

            if im[x, y] != 0:  # boundary point
                theta = findGradient(x, y)
                vectors = table[theta]
                for vector in vectors:
                    acc[vector[0]+x, vector[1]+y] += 1
    return acc

def findMaxima(acc):
    """
    FROM : https://github.com/adl1995/generalised-hough-transform

    :param acc: accumulator array
    :return:
        maxval: maximum value found
        ridx: row index of the maxval
        cidx: column index of the maxval
    """
    ridx, cidx = np.unravel_index(acc.argmax(), acc.shape)
    return [acc[ridx, cidx], ridx, cidx]

