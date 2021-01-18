import pandas as pd
df1 = pd.read_csv('sub1_Bambenek_DGA.csv')          
df2 = pd.read_csv('sub2_Bambenek_DGA.csv')                   
df3 = pd.read_csv('sub3_Bambenek_DGA.csv') 
df4 = pd.read_csv("sub1_netlab_DGA.csv")
df5 = pd.read_csv("sub2_netlab_DGA.csv")
df6 = pd.read_csv("sub3_netlab_DGA.csv")


result = pd.concat([df1, df2, df3, df4, df5, df6])
print(result)
result = result.drop_duplicates("Domain", keep="first")
result = result.sort_values(["Class"])
result.drop(['Unnamed: 2', 'Unnamed: 3'], axis='columns', inplace=True)
print(result.isnull().sum())
print(result)


result_sub1 = result[0:1000000]
result_sub2 = result[1000001:2000001]
result_sub3 = result[2000002:3000001]
result_sub4 = result[3000002:]

result_sub1.to_csv("result_sub1.csv", index=False)
result_sub2.to_csv("result_sub2.csv", index=False)
result_sub3.to_csv("result_sub3.csv", index=False)
result_sub4.to_csv("result_sub4.csv", index=False)