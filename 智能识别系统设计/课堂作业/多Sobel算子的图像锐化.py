import cv2
import numpy as np
import matplotlib.pyplot as plt
# 设置字体为支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

def apply_sobel(image, sobel_x, sobel_y):
    # 获取图像的维度
    height, width = image.shape
    # 创建一个空的图像用于存储结果
    new_image = np.zeros((height, width), dtype=np.float32)
    # 进行卷积操作
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = 0
            gy = 0
            # 计算Gx和Gy
            for k in range(3):
                for l in range(3):
                    gx += image[i + k - 1][j + l - 1] * sobel_x[k][l]
                    gy += image[i + k - 1][j + l - 1] * sobel_y[k][l]
            # 计算梯度幅值 限制范围在0-255
            g = min(255, max(0, ((gx ** 2 + gy ** 2) ** 0.5)))
            new_image[i][j] = g
    return new_image

def sharpen_image(image, sobel_x, sobel_y):
    # 先应用Sobel算子
    edge_image = apply_sobel(image, sobel_x, sobel_y)
    height, width = image.shape
    sharpened_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # 通过加上边缘图像来锐化原图
            # sharpened_image[i][j] = min(255, max(0, image[i][j] + edge_image[i][j]))
            sharpened_image[i][j] = (np.clip(image[i][j]*0.5 + edge_image[i][j]*0.5, 0, 255).astype(np.uint8))
    return sharpened_image

def cv2_sharpen_image(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向
    # 计算梯度幅值
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    # 将梯度幅值限制在0到255之间并转换为uint8
    sobel_magnitude = np.clip(sobel_magnitude, 0, 255).astype(np.uint8)
    # 图像锐化：将边缘信息添加到原图像
    sharpened_image = cv2.addWeighted(image, 0.5, sobel_magnitude, 0.5, 0)
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


image_path = 'test_images/lena_gray.bmp'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = image.astype(np.float32)
sharpened1 = sharpen_image(image, sobel_x, sobel_y)
sharpened2 = sharpen_image(image, sobel_u, sobel_v)

plt.figure(figsize=(12, 6))
plt.subplot(1,3,1)
plt.title("原始图像")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1,3,2)
plt.title("应用sobelx、y图像锐化后的图像")
plt.imshow(sharpened1, cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("应用sobelu、v图像锐化后的图像")
plt.imshow(sharpened2, cmap="gray")
plt.axis("off")

plt.show()