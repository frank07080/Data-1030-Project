Data Project Proposal  
Guanzhong Chen  
Data 1030  
Brown ID: 140268394


# Introductory and Problem


Airbnb stands for "AirBed and Breakfast." More and more people nowadays choose to live in Airbnbs instead of hotels when they are out traveling. There are reasons why Airbnbs become more popular than hotels when it comes to where to live during traveling. One of them is that you are renting houses from a person that give a total different feeling from hotels. The other one may be that Airbnbs are, in a sense, cheaper than hotels.


Price of Airbnb, then, becomes our target variable especially in city like New York. Different from classification problem, predicting the price of Airbnb is a problem of regression because price is a continuous variable. So why is price important? The reason is that most people would to take price as their first consideration when budgets are taken into their accounts. It would be convenient if people can predict the price of an Airbnb if they are given enough information of that Airbnb.


# Description of Dataset


The dataset is from Kaggle called "New York City Airbnb Open Data." There are a total of 48895 data points and 16 features in the dataset. The dataset is well-documented, and we can check a description for each feature in Kaggle.


There are several public projects where data has been used. One of them is called "Maps of NYC Airbnbs with Python." The data was used for finding a listing that meets specific criteria for an upcoming trip of the author. The features were used by filtering out unimportant features and finding specific requirements of an interested feature satisfying the author's criteria. For example, the author would want the host to must have more than 10 reviews.


The other one is called "Data Exploration on NYC Airbnb." The data was used for visualizing and analyzing in that project. For example, the feature host IDs was used for visualizing hosts with most listings in New York city. And other features were used for visualizing purposes too.


# Dataset Preprocessing


There are a total of 3 features we are interested in for now and will be preprocessed.


The first feature is the neighbourhood group in New York city. I have chosen the OrdinalEncoder for this categorical feature. The reason is that the name of neighbourhood group can be ordered in an alphabetical order.

```python
import pandas as pd

# load data from a csv file
df = pd.read_csv('AB_NYC_2019.csv') # there are also pd.read_excel(), and pd.read_sql()
```

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

X_train1 = pd.DataFrame(df['neighbourhood_group'])

# initialize the encoder
enc = OrdinalEncoder(categories = [['Bronx','Brooklyn', 'Manhattan','Queens','Staten Island']]) # The ordered list of 
# fit the training data
enc.fit(X_train1)
# print the categories - not really important because we manually gave the ordered list of categories
print(enc.categories_)
# transform X_train. We could have used enc.fit_transform(X_train) to combine fit and transform
X_train1_oe = enc.transform(X_train1)

print(X_train1_oe[:10])
df['neighbourhood_group'] = X_train1_oe
print(df['neighbourhood_group'][:10])
```

The second feature is the number of reviews per month of an Airbnb in New York city. I have chosen the StandardScaler for this continuous feature. The reason is that these continuous values follow a tail distribution because a small number of the most popular Airbnbs will get the most reviews.

```python
from sklearn.preprocessing import StandardScaler

X_train2 = pd.DataFrame(df['reviews_per_month'])
scaler = StandardScaler()
X_train2_oe = scaler.fit_transform(X_train2)
print(X_train2_oe[:10])

df['reviews_per_month'] = X_train2_oe
print(df['reviews_per_month'][:10])
```

The third feature is the type of room Airbnbs in New York city. I have chosen the OrdinalEncoder for this categorical feature. The reason is that they can be ordered by how good a room is. For example, the entire apartment/room is better than a private room. And a private room is better than a shared room.

```python
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

X_train3 = pd.DataFrame(df['room_type'])

# initialize the encoder
enc = OrdinalEncoder(categories = [['Entire home/apt','Private room', 'Shared room']]) # The ordered list of 
# fit the training data
enc.fit(X_train3)
# print the categories - not really important because we manually gave the ordered list of categories
print(enc.categories_)
# transform X_train. We could have used enc.fit_transform(X_train) to combine fit and transform
X_train3_oe = enc.transform(X_train3)

print(X_train3_oe[:10])
df['room_type'] = X_train3_oe
print(df['room_type'][:10])
```

```python

```
