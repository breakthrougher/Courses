import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


def apply_one_sobel(image, sobel):
    # 获取图像的维度
    height, width = image.shape
    # 创建一个空的图像用于存储结果
    new_image = np.zeros((height, width), dtype=np.float32)
    # 进行卷积操作
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            g = 0
            for k in range(3):
                for l in range(3):
                    g += image[i + k - 1][j + l - 1] * sobel[k][l]
            new_image[i][j] = g
    return new_image


def sharpen_one_image(image, sobel):
    # 先应用Sobel算子
    edge_image = apply_one_sobel(image, sobel)
    height, width = image.shape
    sharpened_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # 通过加上边缘图像来锐化原图
            sharpened_image[i][j] = (np.clip(image[i][j] * 0.5 + edge_image[i][j] * 0.5, 0, 255)).astype(np.uint8)
    return sharpened_image


# 定义Sobel算子
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], dtype=np.float32)
sobel_u = np.array([[-2, -1, 0],
                    [-1, 0, 1],
                    [0, 1, 2]], dtype=np.float32)
sobel_v = np.array([[0, 1, 2],
                    [-1, 0, 1],
                    [-2, -1, 0]], dtype=np.float32)

image_path = 'test_images/peppers.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)

sharpened1 = sharpen_one_image(image, sobel_x)
sharpened2 = sharpen_one_image(image, sobel_y)
sharpened3 = sharpen_one_image(image, sobel_u)
sharpened4 = sharpen_one_image(image, sobel_v)

plt.figure(figsize=(12, 6))
plt.subplot(1, 5, 1)
plt.title("原始图像")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.title("应用sobelx图像锐化后的图像")
plt.imshow(sharpened1, cmap="gray")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.title("应用sobely图像锐化后的图像")
plt.imshow(sharpened2, cmap="gray")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.title("应用sobelu图像锐化后的图像")
plt.imshow(sharpened3, cmap="gray")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.title("应用sobelv图像锐化后的图像")
plt.imshow(sharpened4, cmap="gray")
plt.axis("off")

plt.show()
