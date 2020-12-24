import math
import time
import pickle
import joblib
import itertools
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy
from sklearn import svm, metrics
from matplotlib import pyplot as plt
################################################## 전처리에 필요한 txt ##################################################

with open("googlebooks-eng-10000.txt", "r") as f:
    word_list = [line.rstrip() for line in f if len(line) > 3 ]
f.close() 
with open("TLD_list.txt", "r") as g:
    TLD_list = [line.rstrip() for line in g]
g.close()  

file=open("3gram","rb")
three_gram=pickle.load(file)

file=open("4gram","rb")
four_gram=pickle.load(file)

file=open("5gram","rb")
five_gram=pickle.load(file)


std_scaler = pickle.load(open('StandardScaler2.pkl','rb'))

################################################## 피처 함수 ##################################################

def Entropy(df): # 엔트로피 함수
        entropy = []
        counts = [Counter(i) for i in list(df["Domain"])]
        for domain in counts:
            prob = [ float(domain[c]) / sum(domain.values()) for c in domain ]
            entropy.append(-(sum([ p * math.log(p) / math.log(2.0) for p in prob ]))) 
        return entropy

####################################################### 저장되있는 머신러닝 불러옴 #######################################################
clf_from_joblib = joblib.load('RandomForestClassifier_2000000 v2.pkl') 
####################################################### 새로운 데이터 입력 #######################################################

string_list =  pd.DataFrame({"Domain":[]})
Test_df = pd.DataFrame()

while 1:
    string = input()
    if(string == ""):
        break
    Test_df = string_list.append( {'Domain' : string}, ignore_index=True) 
 ####################################################### 새로운 데이터 전처리 #######################################################
    
    Test_df["TLD"] = list(itertools.chain.from_iterable([[next((tld for i, tld in enumerate(TLD_list) if Domain[-len(tld):] == tld), Domain[Domain.rfind("."):])] for Domain in Test_df.Domain])) # 도메인 TLD
    Test_df["Sub_Domain"] = [Test_df.Domain.str[:-len(Test_df.TLD.str)] for Test_df.Domain.str, Test_df.TLD.str in zip(Test_df.Domain, Test_df.TLD)]
    Test_df["TLD_index"] = list(itertools.chain.from_iterable([[next(((i+1) for i, tld in enumerate(TLD_list) if Domain[-len(tld):] == tld), 0)] for Domain in Test_df.Domain]))
    Test_df["3-gram_Score"] = [(sum([three_gram.get(Domain[i:i+3]) for i in range(len(Domain)-2) if(Domain[i:i+3] in three_gram)]) / len(Domain)) for Domain in Test_df["Sub_Domain"]]
    Test_df["4-gram_Score"] = [(sum([four_gram.get(Domain[i:i+4]) for i in range(len(Domain)-2) if(Domain[i:i+4] in four_gram)]) / len(Domain)) for Domain in Test_df["Sub_Domain"]]
    Test_df["5-gram_Score"] = [(sum([five_gram.get(Domain[i:i+5]) for i in range(len(Domain)-2) if(Domain[i:i+5] in five_gram)]) / len(Domain)) for Domain in Test_df["Sub_Domain"]]
    Test_df["Length"] = Test_df.Sub_Domain.str.len() # 도메인 길이
    Test_df["Numeric_ratio"] = Test_df.Sub_Domain.str.count('[0-9]') / Test_df.Sub_Domain.str.len() # 도메인에 포함된 숫자 개수 / 도메인 길이
    Test_df["Vowel_ratio"] = Test_df.Sub_Domain.str.count('[a,e,i,o,u]') / Test_df.Sub_Domain.str.len() # 도메인에 포함된 모음 개수 / 도메인 길이
    Test_df["Consonant_ratio"] = Test_df.Sub_Domain.str.count('[b,c,d,f,g,h,j,k,l,m,n,p,q,r,s,t,v,w,x,y,z]') / Test_df.Sub_Domain.str.len() # 도메인에 포함된 자음 개수 / 도메인 길이
    Test_df["Consecutive_consonant"] = Test_df.Sub_Domain.str.count('[^.aeiou]{3,}') # 연속되는 3글자(자음) 개수
    Test_df["Consecutive_Vowel"] = Test_df.Sub_Domain.str.count('[aeiou]{2,}') # 연속되는 2글자(모음) 개수
    Test_df["period"] = Test_df.Sub_Domain.str.count('[.]') # . 개수
    Test_df["Entropy"] = Entropy(Test_df)  # Shannon 엔트로피 
    Test_df["Max_Consecutive_Consonant"] = [len(max(i,key=len)) if(len(i) != 0) else 0 for i in Test_df.Sub_Domain.str.findall('[^.aeiou]{3,}')] # 연속되는 3글자(자음,숫자) 최대 값
    Test_df["Max_voewl_Consonant"] = [len(max(i,key=len)) if(len(i) != 0) else 0 for i in Test_df.Sub_Domain.str.findall('[aeiou]{2,}')] # 연속되는 2글자(모음) 최대 값
    Test_df["Meaning_count"] = [len([word for word in word_list if(word in Domain) ]) for Domain in Test_df["Sub_Domain"].to_list()]
    print(Test_df)
    Test_df.to_csv("Test_df.csv", index=False)
 ####################################################### 실시간 데이터 예측 #######################################################
    Test_data = Test_df.iloc[:, 3:]
    test_scaler = std_scaler.transform(Test_data)
    Test_data = pd.DataFrame(test_scaler, columns=Test_data.columns, index=list(Test_data.index.values))
    Test_Predict = clf_from_joblib.predict(Test_data)
    print(Test_data)
    Test_data.to_csv("Test_df2.csv", index=False)
    """
    for name, importance in zip(Test_data.columns, clf_from_joblib.feature_importances_):
        print(name, "=", importance)
        importances = clf_from_joblib.feature_importances_    
        indices = np.argsort(importances)   
        plt.title('Feature Importances', fontsize=24)
        plt.barh(range(len(indices)), importances[indices], color='#8DD8CC', align='center')
        plt.yticks(range(len(indices)), [Test_data.columns[i] for i in indices], fontsize=24)
        plt.xlabel('Relative Importance', fontsize=24)
        plt.show()
    """
    print(string ,Test_Predict)
    string_list = string_list[0:0]
    Test_df = pd.DataFrame()


   
