import math
import time
import pickle
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy.stats import entropy

def Entropy(df):
        entropy = []
        counts = [Counter(i) for i in list(df["Domain"])]
        for domain in counts:
            prob = [ float(domain[c]) / sum(domain.values()) for c in domain ]
            entropy.append(-(sum([ p * math.log(p) / math.log(2.0) for p in prob ]))) 
        return entropy

file=open("3gram","rb")
three_gram=pickle.load(file)

file=open("4gram","rb")
four_gram=pickle.load(file)

file=open("5gram","rb")
five_gram=pickle.load(file)
##################################################  정상, DGA CSV 불러와 Dataframe에 저장   ##################################################
Normal_Train_df = pd.read_csv('Train_Normal.csv')  

#with open("google-books-common-words-lower.txt", "r") as f:
#    lines = [line.rstrip() for line in f if len(line) > 4 ]
#f.close()
with open("TLD_list.txt", "r") as g:
    TLD_list = [line.rstrip() for line in g]
g.close()
#df1 = pd.concat([df1[:500], df1[500:1000].sample(frac=1)]).reset_index(drop=True)

##################################################  정상, DGA의 Dataframe 하나의 Dataframe에 병합    ##################################################
                     #정상, DGA의 Dataframe 하나의 Dataframe에 병합
Train_df = pd.concat( [Normal_Train_df[:100], Normal_Train_df[1000001:].sample(frac=1).reset_index(drop=True).head(100)] ) #frac=1을 설정해서 모든 데이터(100%)를 샘플링,
                                                                                         #reset_index를 해서 기존의 index가 아닌 새로운 indexing                                                                                                                                                                         

for i in range(100, 200):                          # 500 ~ 1000행 까지
    Train_df.iloc[i,0] = i - 99                         # shuffle된 DGA 순서대로 Rank열에 순위 기록                                                                                      
################################################## 통계용 Dataframe에 Label, Rank, Domain, F0(Url_Length 저장)    ##################################################
"""
Train_df["Length"] = Train_df.Domain.str.len() # 도메인 길이
Train_df["Numeric_ratio"] = Train_df.Domain.str.count('[0-9]') / Train_df.Domain.str.len() # 도메인에 포함된 숫자 개수 / 도메인 길이
Train_df["Vowel_ratio"] = Train_df.Domain.str.count('[a,e,i,o,u]') / Train_df.Domain.str.len() # 도메인에 포함된 모음 개수 / 도메인 길이
Train_df["Consonant_ratio"] = Train_df.Domain.str.count('[b,c,d,f,g,h,j,k,l,m,n,p,q,r,s,t,v,w,x,y,z]') / Train_df.Domain.str.len() # 도메인에 포함된 자음 개수 / 도메인 길이
Train_df["Consecutive_consonant"] = Train_df.Domain.str.count('[^.aeiou]{3,}') # 연속되는 3글자(자음) 개수
Train_df["Consecutive_Vowel"] = Train_df.Domain.str.count('[aeiou]{2,}') # 연속되는 2글자(모음) 개수
Train_df["period"] = Train_df.Domain.str.count('[.]') # . 개수
Train_df["Entropy"] = Entropy(Train_df)  # Shannon 엔트로피 
Train_df["Max_Consecutive_Consonant"] = [len(max(i,key=len)) if(len(i) != 0) else 0 for i in Train_df.Domain.str.findall('[^.aeiou]{3,}')] # 연속되는 3글자(자음) 최대 값
Train_df["Max_voewl_Consonant"] = [len(max(i,key=len)) if(len(i) != 0) else 0 for i in Train_df.Domain.str.findall('[aeiou]{2,}')] # 연속되는 3글자(모음) 최대 값
Train_df["Meaning_count"] = [len([word for word in word_list if(word in url) ]) for url in string_list["Domain"].to_list()]
Train_df["TLD"] = list(itertools.chain.from_iterable([[next((tld for i, tld in enumerate(TLD_list) if Domain[-len(tld):] == tld), Domain[Domain.rfind("."):])] for Domain in Train_df.Domain])) # 도메인 TLD
Train_df["Sub_Domain"] = [Train_df.Domain.str[:-len(Train_df.TLD.str)] for Train_df.Domain.str, Train_df.TLD.str in zip(Train_df.Domain, Train_df.TLD)]
Train_df["3-gram_Score"] = [(sum([three_gram.get(Domain[i:i+3]) for i in range(len(Domain)-2) if(Domain[i:i+3] in three_gram)]) / len(Domain)) for Domain in Train_df["Sub_Domain"]]
Train_df["4-gram_Score"] = [(sum([four_gram.get(Domain[i:i+4]) for i in range(len(Domain)-2) if(Domain[i:i+4] in four_gram)]) / len(Domain)) for Domain in Train_df["Sub_Domain"]]
Train_df["5-gram_Score"] = [(sum([five_gram.get(Domain[i:i+5]) for i in range(len(Domain)-2) if(Domain[i:i+5] in five_gram)]) / len(Domain)) for Domain in Train_df["Sub_Domain"]]
"""
sns.scatterplot(x='Rank', # x축 Rank

                y='5-Length', #y축 

                hue='Label', # different colors by group

                #style='species', # different shapes by group

                s=100, # marker size

                data=Train_df) # 소스 데이터
plt.show() # 그래프 출력