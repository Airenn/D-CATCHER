import numpy as np
import pandas as pd
import pickle
import joblib
import lightgbm as lgb 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
################################################## 피처 불러옴 ##################################################
DGA_Train_df = pd.read_csv('C:\\Users\\USER\\Desktop\\박해민\\공부\\K shield\\프로젝트\\코드Train_StandardScaler_2000000.csv') # 미리 전처리 되어있는 피처 CSV 불러옴 
#DGA_Test_df = pd.read_csv('Test_StandardScaler_2000000.csv')
#Test_df = pd.concat( [DGA_Test_df[:].sample(frac=1).reset_index(drop=True).head(100000)] )
#Train_data = DGA_Train_df.iloc[:, 5:] # 사용할 피처 열 
#Train_label = DGA_Train_df.iloc[:, 0] # Label

Test_data = DGA_Train_df.iloc[:, 5:] # 사용할 피처 열 
Test_label = pd.DataFrame(d
ata=DGA_Train_df.Label, columns=['Label'])
encoder = LabelEncoder()
encoder.fit(Test_label)
labels = encoder.transform(Test_label)

train__data, test__data, train__label,test__label = train_test_split(Test_data,labels,random_state=2020, shuffle=True) # Train_df.csv에서 학습, 검증용 데이터셋 나눔
################################################## 머신러닝 실행,학습,예측 ##################################################

#clf = svm.SVC(gamma="auto", verbose=2) # 모델 실행 이 부분 다른 알고리즘들로 바꿔가면서 사용
#clf_from_joblib = joblib.load('RandomForestClassifier_2000000.pkl') 

train_ds = lgb.Dataset(train__data, label= train__label)
test_ds = lgb.Dataset(test__data, label= test__label)

params = {'learning_rate' : 0.01,
            'max_depth' : 16,
            'boosting' : 'gbdt',
            'objective' : 'binary',
            'metric' : 'binary_logloss',
            'is_training_metric' : True,
            'num_leaves' : 144,
            'feature_fraction' : 0.9,
            'bagging_fraction' : 0.7,
            'bagging_freq' : 5,
            'seed' : 2020
}

lgb_model = lgb.train(params, train_ds, 1000, train_ds, verbose_eval=100, early_stopping_rounds=100)
#lgb.fit(train__data, train__label) # Train_df.csv에서 나눠진 학습용 데이터 학습

####################################################### 정답 예측 #######################################################
Test_Predict = lgb_model.predict(test__data) # 검증용 데이터셋 정답 예측
#Test_Predict = clf_from_joblib.predict(Test_data)
####################################################### 결과 분석 #######################################################
#Test_Predict = pd.DataFrame(Test_Predict, columns=Test_label.columns, index=list(Test_label.index.values))

print(Test_Predict)
print(type(Test_Predict))

print(test__label)
print(type(test__label))

prediction=np.expm1(Test_Predict)
Test_Predict=prediction.astype(int)

print("테스트 세트의 정확도: {:.2f}".format(np.mean(Test_Predict == test__label)))
#print("테스트 세트의 정확도: {:.2f}".format(np.mean(Test_Predict == Test_label)))
confusion = confusion_matrix(test__label,Test_Predict, labels=(0, 1))
print("오차 행렬:\n{}".format(confusion))
print("정밀도(precision) = ", precision_score(test__label,Test_Predict, pos_label=0))
print("정답률(accuracy) = ", metrics.accuracy_score(test__label,Test_Predict))
print("재현율(Recall) = ", recall_score(test__label,Test_Predict, average="binary", pos_label=0))
report = classification_report(test__label,Test_Predict, digits=4)
print(report)
print("End")
