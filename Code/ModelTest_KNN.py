import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
################################################## 피처 불러옴 ##################################################
DGA_Train_df = pd.read_csv('Train_StandardScaler_2000000.csv') # 미리 전처리 되어있는 피처 CSV 불러옴 
#DGA_Test_df = pd.read_csv('Test_StandardScaler_200000 v2.csv')
Train_data = DGA_Train_df.iloc[:, 5:] # 사용할 피처 열 
Train_label = DGA_Train_df.iloc[:, 0] # Label

train__data, test__data, train__label,test__label = train_test_split(Train_data,Train_label,random_state=2020, shuffle=True) # Train_df.csv에서 학습, 검증용 데이터셋 나눔
################################################## 머신러닝 실행,학습,예측 ##################################################

clf = KNeighborsClassifier(n_neighbors=5)  # 모델 실행 이 부분 다른 알고리즘들로 바꿔가면서 사용
clf.fit(train__data, train__label) # Train_df.csv에서 나눠진 학습용 데이터 학습
####################################################### 정답 예측 #######################################################
Test_Predict = clf.predict(test__data) # 검증용 데이터셋 정답 예측
#Test_Predict = clf_from_joblib.predict(Test_data)
####################################################### 결과 분석 #######################################################

print("테스트 세트의 정확도: {:.2f}".format(np.mean(Test_Predict == test__label)))
#print("테스트 세트의 정확도: {:.2f}".format(np.mean(Test_Predict == Test_label)))
confusion = confusion_matrix(test__label,Test_Predict, labels=("DGA", "Normal"))
print("오차 행렬:\n{}".format(confusion))
print("정밀도(precision) = ", precision_score(test__label,Test_Predict, pos_label="DGA"))
print("정답률(accuracy) = ", metrics.accuracy_score(test__label,Test_Predict))
print("재현율(Recall) = ", recall_score(test__label,Test_Predict, average="binary", pos_label="DGA"))
report = classification_report(test__label,Test_Predict, digits=4)
print(report)
print("End")
