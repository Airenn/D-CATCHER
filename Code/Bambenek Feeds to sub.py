import pandas as pd

df = pd.read_csv('Total_Bambenek_DGA.csv')          

df.drop(['Unnamed: 2',  'Unnamed: 3',  'Clas'], axis='columns', inplace=True)
df.drop([2885044], inplace=True)
print(df.isnull().sum())
print(df)
#df = pd.concat([df1, df2, df3, df4, df5, df6])
df = df.drop_duplicates("Domain", keep="first")
df = df.sort_values(["Class"])
print(df)

#df['count'] = df.groupby('name')['name'].transform('count')
sub1 = df[:1000000]
sub2 = df[1000000:2000000]
sub3 = df[2000000:]
df.to_csv("Total_Bambenek_DGA2.csv", index=False)

sub1.to_csv("sub1_Bambenek_DGA.csv", index=False)
sub2.to_csv("sub2_Bambenek_DGA.csv", index=False)
sub3.to_csv("sub3_Bambenek_DGA.csv", index=False)