import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Original image
image = cv.imread("./Resources/Photos/cats.jpg")
cv.imshow("Original", image)

# Gray scale histogram
gray_image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
cv.imshow("Gray", gray_image)

gray_hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])

# Masked histogram
blank = np.zeros(image.shape[:2], dtype="uint8")
circle_mask = cv.circle(
    blank.copy(), (image.shape[1] // 2 + 170, image.shape[0] // 2 + 100), 100, 255, -1
)
masked_gray = cv.bitwise_and(gray_image, gray_image, mask=circle_mask)
masked = cv.bitwise_and(image, image, mask=circle_mask)
cv.imshow("Masked Gray", masked_gray)
cv.imshow("Masked", masked)
masked_gray_hist = cv.calcHist([gray_image], [0], circle_mask, [256], [0, 256])

plt.figure()
plt.title("Grayscale Histogram")
plt.ylabel("# of pixesl")
plt.xlabel("Bins")
plt.plot(gray_hist, "-k")
plt.plot(masked_gray_hist, "--k")
plt.legend(["Grayscale", "Masked Grayscale"])
plt.xlim([0, 256])
plt.pause(0.001)  # workaround for non blocking plot

# BGR histograms
bgr = cv.split(image)  # split image by channels

b_histo = cv.calcHist(bgr, [0], None, [256], [0, 256])
g_histo = cv.calcHist(bgr, [1], None, [256], [0, 256])
r_histo = cv.calcHist(bgr, [2], None, [256], [0, 256])
masked_b_histo = cv.calcHist(bgr, [0], circle_mask, [256], [0, 256])
masked_g_histo = cv.calcHist(bgr, [1], circle_mask, [256], [0, 256])
masked_r_histo = cv.calcHist(bgr, [2], circle_mask, [256], [0, 256])

plt.figure()
plt.title("BGR Histogram")
plt.ylabel("# of pixesl")
plt.xlabel("Bins")
plt.plot(b_histo, "b")
plt.plot(g_histo, "g")
plt.plot(r_histo, "r")
plt.plot(masked_b_histo, "--b")
plt.plot(masked_g_histo, "--g")
plt.plot(masked_r_histo, "--r")
plt.legend(["Blue", "Green", "Red", "Masked_Blue", "Masked_Green", "Masked_Red"])
plt.xlim([0, 256])
plt.pause(0.001)  # workaround for non blocking plot


cv.waitKey(0)
cv.destroyAllWindows()
