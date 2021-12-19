# -*- coding: UTF-8 -*-
import pyautogui
import time
from cnocr import CnOcr
import cv2
from PIL import ImageGrab, Image
import numpy as np

pyautogui.PAUSE = 1


class Fish:
    def findfish(self, res):
        for line in res:
            if line[0] == ['浮', '漂', '：', '溅', '起', '水', '花']:
                return True

        return False

    def fish(self, img):
        ocr = CnOcr()

        textImg = self.detect(img)
        # ii = Image.fromarray(textImg.astype('uint8')).convert('RGB')
        # ii.save(str(time.time()) + '.jpg')

        # cnocr识别文本
        print(textImg.shape)
        res = ocr.ocr(textImg)
        print("Predicted Chars:", res)

        if self.findfish(res):
            pyautogui.click(button='right')
            pyautogui.click(button='right')
            time.sleep(1)
        else:
            time.sleep(0.5)

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dilation = self.preprocess(gray)
        x, y, w, h = self.findTextRegion(dilation)

        ii = img[y:y + h, x:x + w]
        gray_ii = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
        _, binary_ii = cv2.threshold(gray_ii, 60, 255, cv2.THRESH_BINARY)

        return binary_ii

    def preprocess(self, gray):
        # Sobel算子
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))
        dilation = cv2.dilate(binary, element2, iterations=1)
        erosion = cv2.erode(dilation, element1, iterations=1)
        dilation2 = cv2.dilate(erosion, element2, iterations=4)

        return dilation2

    def findTextRegion(self, img):
        # 查找轮廓
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        maxArea = 0
        maxContour = 0
        if len(contours) == 0:
            return 0, 0, 0, 0
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area > maxArea:
                maxArea = area
                maxContour = cnt
        x, y, w, h = cv2.boundingRect(maxContour)

        return x, y, w, h


if __name__ == '__main__':
    fish = Fish()
    img = Image.open('./img/110.jpg')
    # img = Image.open('./img/110.jpg')
    img.save('00.jpg')
    img = np.array(img)

    ii = fish.detect(img)
    ii = Image.fromarray(ii.astype('uint8')).convert('RGB')
    ii.save('11.jpg')


