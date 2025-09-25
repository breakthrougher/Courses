import cv2
import numpy as np
import os
from sklearn.decomposition import PCA as Sklearn_PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

component = 27
Neighbors = 21
# 数据格式为按列
def PCA(X_train, components):
    # Prepare
    col = X_train.shape[1]
    # axis=0 表示操作是针对行的方向，但实际上是在处理特征（列）
    Mean = np.mean(X_train, axis=1)
    # X = X_train - Mean
    # 减去每个特征的均值,使得每个特征的均值均为0
    X = X_train - Mean[:, np.newaxis]
    # Calculate eig_vectors, eig_values
    cov = np.dot(X, X.T) / col
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    # 确保特征值、特征向量为实数
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # 对特征值、特征向量进行排序
    index = np.argsort(eigenvalues)[::-1]
    # 大小为样本数量*样本数量
    W = eigenvectors[:, index]

    # Build model
    model = {
        'mean': Mean,
        'W': W[:, :components],
        'components': components
    }

    return model

# 加载人脸检测模型
# face_mode = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt(2).xml')
face_mode = cv2.CascadeClassifier("D:/tmp_data/haarcascade_frontalface_alt(2).xml")

# 训练集和测试集
train_path = "train_samples1"
train_images = []
train_labels = []

# 人脸检测
for label in os.listdir(train_path):
    label_path = os.path.join(train_path, label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_mode.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=Neighbors)
            print(f"Image: {img_name}, Detected faces: {len(faces)}")
            for (x, y, w, h) in faces:
                face = gray_img[y:y+h, x:x+w]
                face_sized = cv2.resize(face, (50, 50))
                train_images.append(face_sized.flatten())
                train_labels.append(label)

# 将训练数据转为numpy数组
X_train = np.array(train_images)
y_train = np.array(train_labels)

print(f"Detected Train faces num are: {X_train.shape[0]}/100")

# sklearn库中的PCA进行特征提取
# pca = Sklearn_PCA(n_components=component)
# x_train_pca = pca.fit_transform(X_train)
# project_matrix = pca.components_

# 使用自定义PCA进行特征提取
model = PCA(X_train.transpose(), components=component)
x_train_pca = X_train.dot(model['W'])
print(x_train_pca.shape)
# 读取测试集
test_images = []
test_labels = []
test_path = "test_samples1"
test_num = 0
faces_num = 0
for label in os.listdir(test_path):
    label_path = os.path.join(test_path, label)
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_mode.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=Neighbors)
            for (x, y, w, h) in faces:
                face = gray_img[y:y+h, x:x+w]
                face_sized = cv2.resize(face, (50, 50))
                test_images.append(face_sized.flatten())
                test_labels.append(label)
                faces_num += 1
        test_num += 1
print(f"Detected Test faces num are: {faces_num}/{test_num}")

# 将测试数据转为numpy数组
X_test = np.array(test_images)
y_test = np.array(test_labels)

# KNN训练和识别
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_pca, y_train)

# x_test_pca = pca.transform(X_test)
x_test_pca = X_test.dot(model['W'])

predictions = knn.predict(x_test_pca)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy is {accuracy*100:.2f}")

cv2.destroyAllWindows()
