# import data modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
df =pd.read_csv("Housing_Dataset_Sample.csv")

df.head(n=10)
df.info()
df.describe().T

sns.distplot(df['Price'])
sns.displot(df['Price'])

sns.jointplot(x=df['Avg. Area Income'],y=df['Price'])
sns.pairplot(df)
#列,欄
#: 什麼”到”什麼(至,~)
#絕大多數的Python語法:包含頭，不包含尾
X = df.iloc[:, :5]
y = df['Price']

#切分訓練集與測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=54)



#建立線性回歸模型
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

#預測
predictions = reg.predict(X_test)

#評估模型
from sklearn.metrics import r2_score
r2_score(y_test, predictions)
