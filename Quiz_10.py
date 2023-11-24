# 필요한 라이브러리 import
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# 데이터 불러오기
file_path = "/Users/judongcheol/Library/Mobile Documents/com~apple~CloudDocs/대학교/2학년 2학기/프로그래밍 2/코딩 파일/dongcheolju_Quiz_10/09_irisdata.csv"
column_name = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(file_path, names=column_name)

print(dataset.columns)

print(dataset.shape)

print(dataset.describe())

print(dataset.groupby('class').size())

# scatter_matrix 그래프 저장
scatter_matrix(dataset, figsize=(10, 10))
plt.savefig("scatter_matrix.png")

# 독립변수 X와 종속변수 Y로 분할
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

model = DecisionTreeClassifier()

# K-fold(10개의 폴드 지정) 및 cross validation(평가 지표 accuracy)
kfold = KFold(n_splits=10, random_state=10, shuffle=True)
results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

model.predict()

print(results.mean())
