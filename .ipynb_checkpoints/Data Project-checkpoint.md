Data Project Proposal  
Guanzhong Chen  
Data 1030  
Brown ID: 140268394  
GitHub Link: https://github.com/frank07080/Data-1030-Project


# Introductory and Problem


Airbnb stands for "AirBed and Breakfast." More and more people nowadays choose to live in Airbnbs instead of hotels when they are out traveling. There are reasons why Airbnbs become more popular than hotels when it comes to where to live during traveling. One of them is that you are renting houses from a person that give a total different feeling from hotels. The other one may be that Airbnbs are, in a sense, cheaper than hotels.


Price of Airbnb, then, becomes our target variable especially in city like New York. Different from classification problem, predicting the price of Airbnb is a problem of regression because price is a continuous variable. So why is price important? The reason is that most people would to take price as their first consideration when budgets are taken into their accounts. It would be convenient if people can predict the price of an Airbnb if they are given enough information of that Airbnb.


# Description of Dataset


The dataset is from Kaggle called "New York City Airbnb Open Data." There are a total of 48895 data points and 16 features in the dataset. The dataset is well-documented, and we can check a description for each feature in Kaggle.


There are several public projects where data has been used. One of them is called "Maps of NYC Airbnbs with Python." The data was used for finding a listing that meets specific criteria for an upcoming trip of the author. The features were used by filtering out unimportant features and finding specific requirements of an interested feature satisfying the author's criteria. For example, the author would want the host to must have more than 10 reviews.


The other one is called "Data Exploration on NYC Airbnb." The data was used for visualizing and analyzing in that project. For example, the feature host IDs was used for visualizing hosts with most listings in New York city. And other features were used for visualizing purposes too.


# Dataset Preprocessing


There are a total of 3 features we are interested in for now and will be preprocessed.


The first feature is the neighbourhood group in New York city. I have chosen the OneHotEncoder for this categorical feature. The reason is that there is no specific order of the neighbourhood groups. The first 10 rows of this feature is shown below.


![](figures/f1.PNG)


The next feature is room type of Airbnbs in New York city. I have chosen the OrdinalEncoder for this categorical feature. The reason is that they can be ordered by how good a room is. For example, the entire apartment/room is better than a private room. And a private room is better than a shared room. The first 10 rows of this feature is shown below.


![](figures/f2.PNG)


The next feature is number of reviews of Airbnbs in New York city. I have chosen the Standard Scaler for this continuous feature. The reason is that some of good Airbnbs will get significantly more reviews than the others. And this follows a tailed distribution. The first 10 rows of this feature is shown below.


![](figures/f3.PNG)


The next feature is reviews per month of Airbnbs in New York city. This is very similar to the last feature. I have chosen the Standard Scaler for this continuous feature. The reason is the same a s the last one. The first 10 rows of this feature is shown below.


![](figures/f4.PNG)


The next feature is availability of a year of Airbnbs in New York city. I have chosen the MinMax Scaler for this continuous feature. The reason is that there is a clear min and max of this feature. The first 10 rows of this feature is shown below.


![](figures/f5.PNG)

```python

```
