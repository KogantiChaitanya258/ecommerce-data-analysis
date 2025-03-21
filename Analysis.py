import pandas as pd

import numpy as np
df=pd.read_csv('ecommerce_data.csv')
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

#unique Products and customers
# print(df["customer_id"].unique().sum())
# print(df["product_id"].unique().sum())
df.fillna(df.median(numeric_only=True),inplace=True)
df.fillna(df.mode().iloc[0],inplace=True)
df["order_date"]=pd.to_datetime(df['order_date'])
df["customer_id"]=df['customer_id'].astype(str)
df["product_id"]=df['product_id'].astype(str)
df.drop_duplicates(inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8,5))
sns.histplot(df['purchase_amount'],bins=50,kde=True)
plt.title("Purchase Amount Analysis")
#plt.show()

top_categories=df.groupby('category')['purchase_amount'].sum().sort_values(ascending=True)
plt.figure(figsize=(10,5))
sns.barplot(x=top_categories.index,y=top_categories.values)
plt.xticks(rotation=45)
plt.title("Top 10 Best-Selling Product Categories")
#plt.show()

df['month']=df['order_date'].dt.month
monthly_sales=df.groupby('month')['purchase_amount'].sum()
plt.figure(figsize=(10,5))
sns.lineplot(x=monthly_sales.index,y=monthly_sales.values)
plt.title("Monthly Sales Analysis")
plt.xlabel("Month")
plt.ylabel("Total Sales")
#plt.show()

df["day_of_week"] = df["order_date"].dt.day_name()
df["day_of_week"].value_counts().max()
df["is_weekend"]=df["day_of_week"].isin(['Saturday','Sunday']).astype(int)

customer_value=df.groupby('customer_id')['purchase_amount'].sum().reset_index()
customer_value.rename(columns={'purchase_amount':'customer_lifetime_value'}, inplace=True)
df=df.merge(customer_value,on="customer_id",how="left")

df['discount_effect']=df['discount_rate']*df['purchase_amount']
top_spenders=df.groupby('customer_id')['purchase_amount'].sum().nlargest(5)
#print(top_spenders)

#Clustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

customer_features=df.groupby('customer_id')[["purchase_amount","discount_rate"]].mean()
#print(customer_features)
scaler= StandardScaler()
scaled_features=scaler.fit_transform(customer_features)
#print(scaled_features)

kmeans=KMeans(n_clusters=3,random_state=42)
customer_features['segment']=kmeans.fit_predict(scaled_features)
print(customer_features)
plt.figure(figsize=(10,5))
sns.scatterplot(x=customer_features['purchase_amount'],y=customer_features['discount_rate'],hue=customer_features['segment'])
plt.show()


from statsmodels.tsa.arima.model import ARIMA

daily_sales = df.groupby("order_date")["purchase_amount"].sum()
model = ARIMA(daily_sales, order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

plt.figure(figsize=(10,5))
#plt.plot(daily_sales, label="Historical Sales")
plt.plot(pd.date_range(daily_sales.index[-1], periods=30, freq='D'), forecast, label="Forecast", color="red")
plt.legend()
plt.title("30-Day Sales Forecast")
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = df[["product_price", "discount_rate", "customer_lifetime_value"]]
y = df["purchase_amount"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse}")