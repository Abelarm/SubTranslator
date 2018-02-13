import cv2

import numpy as np
import os.path
import itertools

#import pytesseract
#from PIL import Image

filename = "./dataset/squaresub.jpg"


def get_contours(image, bound, comparator):

    blur = cv2.GaussianBlur(image, (9, 9), 0)
    edges = cv2.Canny(blur, 100, 200)

    # cv2.imshow('Canny_Blur', edges)

    contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # print(len(contours))
    good_contours = []

    for i in range(len(contours)):
        flag = False
        for p in contours[i]:
            if comparator(p[0, 1], bound[1]):
                flag = True
                break
        if not flag:
            good_contours.append(contours[i])

    if len(good_contours) == 0:
        return good_contours

    cv2.drawContours(image, good_contours, -1, (0, 255, 255))
    # cv2.imshow(image, "what2")

    merged = []
    centroids = []
    for cnt in good_contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append((cx, cy))

    product = set()

    for element in itertools.product(centroids, centroids):
        if element[0] != element[1]:
            product.add(element)

    print("Number of distances: {}".format(len(product)))

    distances = [np.linalg.norm(np.array(e[0])-np.array(e[1])) for e in product]
    mean_distance = np.mean(distances)

    print("Mean distance: {}".format(mean_distance))

    for i, g_c in enumerate(good_contours):
        test_image = image.copy()
        x, y, w, h = cv2.boundingRect(g_c)

        cv2.rectangle(test_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        mesure = 0

        if w > h:
            mesure = (float(w-h))/w
            if mesure > 0.8:
                print("########FUNDED TOO BIG########")
                good_contours.pop(i)

        if h > w:
            mesure = (float(h-w))/h
            if mesure > 0.8:
                print("########FUNDED TOO BIG########")
                good_contours.pop(i)

        M = cv2.moments(g_c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            dist = 0
            i = 0
            centroid = np.array([cx, cy])
            for c in centroids:
                if list(centroid) != list(c):
                    dist += np.linalg.norm(centroid - np.array(c))
                    i += 1
            if i > 0:
                dist /= i
                mesure = (float(dist-mean_distance))/dist

        if mesure > 0.5:
            print("########FUNDED TOO DISTANT########")
            good_contours.pop(i)

        for x in g_c:
            merged.append(x)

    return merged


def retrive_smallest_rect(image, contours):

    merged = np.array(contours)
    hull = cv2.convexHull(merged)

    x, y, w, h = cv2.boundingRect(hull)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    bigx = x + w
    bigy = y + h
    cutted_image = startimage[y:bigy, x:bigx]

    # cv2.imshow('Hull contours', image)
    # cv2.imshow('Partial result', cutted_image)

    canny = cv2.Canny(cutted_image, 200, 500)
    # cv2.imshow('SUBCANNY', canny)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(canny, kernel)
    contours = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=-1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)
    # cv2.imshow("mask", mask)

    merged = [gg_cc for g_c in contours for gg_cc in g_c]
    merged = np.array(merged)
    hull = cv2.convexHull(merged)

    x, y, w, h = cv2.boundingRect(hull)
    upper_x = x + w
    upper_y = y + h
    # print(mask.shape)
    # print(cutted_image.shape)

    maksed = cv2.bitwise_and(cutted_image, cutted_image, mask=mask)
    final_image = maksed[y:upper_y, x:upper_x]
    # cv2.imshow("masked", final_image)
    # print(final_final.shape)
    final_tresh = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    _, final_tresh = cv2.threshold(final_tresh, 135, 255, cv2.THRESH_BINARY)
    # final_tresh = cv2.GaussianBlur(final_tresh, (3, 3), 1, 1)

    return final_tresh


if os.path.isfile(filename):

    image = cv2.imread(filename, flags=cv2.IMREAD_COLOR)
    cv2.imshow(filename.split('/')[-1], image)
    startimage = image.copy()

    lower_bound = (0, int(image.shape[0]*0.80))
    upper_bound = (0, int(image.shape[0]*0.20))

    cv2.circle(image, lower_bound, 5, (0, 255, 0), 2)
    # cv2.imshow('low_bound', image)
    cv2.circle(image, upper_bound, 5, (0, 255, 0), 2)
    # cv2.imshow('bounds', image)

    contours_up = get_contours(image, upper_bound, lambda x, y: x > y)
    contours_down = get_contours(image, lower_bound, lambda x, y: x < y)

    if len(contours_up) == 0 and len(contours_down) > 0:
        contours = contours_down
    elif len(contours_up) > 0 and len(contours_down) == 0:
        contours = contours_up
    else:
        contours = contours_down

    final_tresh = retrive_smallest_rect(image, contours)

    cv2.imshow("final_final", final_tresh)
    cv2.imwrite('Test1.png', final_tresh)

    # img = Image.fromarray(final)
    # print(pytesseract.image_to_string(img, 'eng'))

    final = final_tresh

    # print(final.shape)

    def thresholdCount(x):
        val = 0
        for v in x:
            if v > 150:
                val += 1
        return val

    hi = np.apply_along_axis(thresholdCount, 0, final)
    hi = hi.reshape((final.shape[1], 1))
    index = np.arange(final.shape[1])
    '''
    plt.bar(index, hi)
    plt.show()
    plt.close()
    '''

    def check_word_spacing(vet, i):
        distance = 5
        subvet = vet[i:i+1]
        #ret = reduce(lambda s: s == 0, subvet, True)
        ret = all([x[0] == 0 for x in subvet])
        return ret

    words = []
    tmp = []

    for i, v in enumerate(hi):
        if len(tmp) == 0:
            if v[0] > 0:
                tmp.append(i)
        else:
            if v[0] == 0 and check_word_spacing(hi, i):
                tmp.append(i)
                words.append(tmp)
                tmp = []

    segmented = cv2.cvtColor(final, cv2.COLOR_GRAY2RGB)

    for l in words:
        # print(l[0],, 0,final.shape[1])
        # print(0,l[0], l[1],final.shape[0])
        letter = final[0:final.shape[0], l[0]:(l[1]+1)]
        # print(letter.shape)
        # cv2.imshow("letter?", letter)
        # cv2.waitKey(0)
        for v in l:
            cv2.line(segmented, (v, 0), (v, final.shape[0]), (0, 255, 255), 1)

    cv2.imshow('segmented?', segmented)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

