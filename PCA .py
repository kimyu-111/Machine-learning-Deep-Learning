import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.cluster import KMeans

# 1. 과일 그리기 함수
def draw_fruits(arr, ratio=1):
    n = len(arr)
    rows = int(np.ceil(n/10))
    cols = n if rows < 2 else 10
    fig, axs = plt.subplots(rows, cols, figsize=(cols*ratio, rows*ratio), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            if i*10 + j < n:
                axs[i, j].imshow(arr[i*10 + j], cmap='gray_r')
            axs[i, j].axis('off')
    plt.show()

# 2. 데이터 준비
url = 'http://bit.ly/fruits_300_data'
filename = 'fruits_300.npy'
# 이미 파일이 있다면 다운로드 생략 가능, 없으면 다운로드
try:
    fruits = np.load(filename)
except:
    urllib.request.urlretrieve(url, filename)
    fruits = np.load(filename)

fruits_2d = fruits.reshape(-1, 100*100)

# 3. PCA (주성분 분석)
# [수정 1] n_compoenents -> n_components (오타 수정)
pca = PCA(n_components=50)
pca.fit(fruits_2d)
print("주성분 형태:", pca.components_.shape)

draw_fruits(pca.components_.reshape(-1, 100, 100))

print("원본 데이터 형태:", fruits_2d.shape)
fruits_pca = pca.transform(fruits_2d)
print("축소된 데이터 형태:", fruits_pca.shape)

# 4. 데이터 복원 및 확인
fruits_inverse = pca.inverse_transform(fruits_pca)
print("복원된 데이터 형태:", fruits_inverse.shape)

fruits_reconstruct = fruits_inverse.reshape(-1, 100, 100)
for start in [0, 100, 200]:
    draw_fruits(fruits_reconstruct[start:start+100])
    print("\n")

print("설명된 분산 비율 합:", np.sum(pca.explained_variance_ratio_))


# 5. 로지스틱 회귀로 성능 비교
lr = LogisticRegression()
target = np.array([0]*100 + [1]*100 + [2]*100)

print("--- 원본 데이터 점수 ---")
scores = cross_validate(lr, fruits_2d, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

print("--- PCA 데이터 점수 ---")
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

# 6. 설명된 분산의 비율(0.5)로 PCA 다시 수행
pca = PCA(n_components=0.5)
pca.fit(fruits_2d)
print("선택된 주성분 개수:", pca.n_components_)

fruits_pca = pca.transform(fruits_2d)
print(fruits_pca.shape)

print("--- PCA(0.5) 데이터 점수 ---")
scores = cross_validate(lr, fruits_pca, target)
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))


# 7. K-Means 군집화
km = KMeans(n_clusters=3, random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_, return_counts=True))

# 군집 결과 이미지 확인
for label in range(0, 3):
    draw_fruits(fruits[km.labels_ == label])
    print("\n")


for label in range(0, 3):
    data = fruits_pca[km.labels_ == label]
    plt.scatter(data[:, 0], data[:, 1])


plt.legend(['apple', 'banana', 'pineapple'])
plt.show()