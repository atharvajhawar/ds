#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business Analytic

# # Name : ATHARVA JHAWAR

# # Task 1 : Prediction using Supervised ML

# # Objective :-

# ### Predict the percentage of an student based on the no. of study hours.</b>

# ### Importing all libraries required in this notebook
# 

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 


# Reading data from remote link
# 

# In[2]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data Successfully loaded")


# In[3]:


#Reading Dataset
df


# In[4]:


#checking the shape of the dataset
df.shape


# In[5]:


#Reading the first 10 observation
df.head(10)


# In[6]:


#Reading the first 10 observation
df.head(10)


# In[7]:


#checking numerical data
df.describe()


# In[8]:


#checking correlation between columns
df.corr()


# In[9]:


#checking the null values in the dataset
df.isnull().sum()


# ##### Now,let us plot a graph using matplotlib to understand the relation between columns.

# In[10]:


# Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='o') 
plt.title('Hours vs Percentage') 
plt.xlabel('Hours Studied') 
plt.ylabel('Percentage Score') 
plt.show()


# # From the graph above, we can clearly see that there is a positive linear relation between
# the number of hours studied and percentage of score.
# Hence we can say that the Percentage scores increases as the number of studied hours
# increases

# ### Preparing the data
# We are going to divide this dataset columns into "attributes"(inputs) and "labels"
# (Outputs).
# 

# In[11]:


X = df.iloc[:, :-1].values 
y = df.iloc[:, 1].values


# In[12]:


X


# In[13]:


y


# In[14]:


# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=0)


# In[15]:


X_train


# In[16]:


X_test


# In[17]:


y_train


# ### Training our model and implementing linear regression algorithm

# In[20]:


LinearModel = LinearRegression() 
LinearModel.fit(X_train, y_train)


# In[21]:


#Ploting the linear regression line
line = LinearModel.coef_*X+LinearModel.intercept_
# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line, color = "yellow");
plt.xlabel("Hours");
plt.ylabel("Scores");
plt.title("Linear Regression Line")
plt.show()


# ### Testing our linear Regression Model
# 

# In[22]:


print(X_test) # Testing data - In Hours
y_pred = LinearModel.predict(X_test) # Predicting the scores


# In[24]:


compare= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
compare


# In[25]:


# Plotting the graph for actual vs predicted values
compare.plot()


# ### What will be predicted score if a student studies for 9.25 hrs/ day?

# In[26]:


# You can also test with your own data
hours = 9.25
own_pred = LinearModel.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ##### Hence we can concluded that if a student is involved in 9.25 hours per day , then
# there is a possibility that the percentage comes out to be 93.69173248737539.

# ### Evaluation Of Our Linear Regression Model
# The final step is to evaluate the performance of algorithm. This step is particularlyimportant to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics
# 
# 

# In[27]:


from sklearn import metrics 
print('Mean Absolute Error:',
 metrics.mean_absolute_error(y_test, y_pred))

