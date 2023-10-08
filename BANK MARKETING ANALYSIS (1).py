#!/usr/bin/env python
# coding: utf-8

# # Importing the required libraries

# In[244]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import time
import seaborn as sns



import warnings
warnings.filterwarnings(action="ignore")


# ### Loading the data into a dataframe 

# In[245]:


data = pd.read_csv('bank-full.csv', delimiter= ';')
data.head(10)


# ### Shape of the dataset

# In[246]:


data.shape


# ### Description of the data

# In[247]:


data.describe(include='all')


# In[248]:


data.dtypes


# In[249]:


data.isnull().sum()


# #### Finding the counts of each column to get a better under

# In[250]:


data['job'].value_counts()


# In[251]:


data['education'].value_counts()


# In[252]:


data['contact'].value_counts()


# In[253]:


data['default'].value_counts()


# In[254]:


data['housing'].value_counts()


# In[255]:


data['loan'].value_counts()


# In[256]:


data['month'].value_counts()


# In[257]:


data['poutcome'].value_counts()


# In[258]:


data['y'].value_counts()


# ### after analyzing the dataset I found that there are many unknown values in the dataset
# ### so I'm gonna remove columns with more number of unknown values and remove all the rows with unkown values

# In[259]:


to_drop = ['contact','poutcome']
data = data.drop(columns =to_drop)
data.head(10)


# ### Removing the rows with unknown values

# In[260]:


target_word = 'unknown'

data = data[~data.apply(lambda row: row.astype(str).str.contains('unknown').any(), axis=1)]

data.head(10)


# ###  The shape of the data After removal of the rows with unknown values.

# In[261]:


data.shape


# In[262]:


# Create a distribution plot for the 'age' column
plt.figure(figsize=(8, 6))  # Set the figure size (optional)
sns.histplot(data=data, x='age', kde=True)  # Create the distribution plot
plt.title('Age Distribution Plot')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# ##### Relationship between the age and subcription

# In[263]:


plt.figure(figsize=(15,10))
ax=sns.countplot(data=data, x='age', hue=data.y)
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.xlabel("age")
plt.title("age Vs subscription")
plt.legend(['no', 'yes'])
plt.show()


# ##  checking for outliers 

# Now let as put a boxplot for the numerical columns so that we can find and remove the outliers , which potentially affects our model accuracy and precision

# In[264]:


# Selecting numeric columns from the DataFrame
num_cols = data.select_dtypes(include='number')

# Creating box plots for all numeric columns
plt.figure(figsize=(12, 6))  # Set the figure size (optional)
sns.boxplot(data=num_cols, orient='v')  # 'orient' can be 'v' for vertical or 'h' for horizontal
plt.title('Box Plots of Numeric Columns')
plt.xlabel('Values')
plt.show()


# #### As we can see there are more number of outliers in the balance and duration columns , so we are gonna remove those outliers

# In[265]:


# Method for identifying outliers for the 'balance' column
Q1_balance = data['balance'].quantile(0.25)
Q3_balance = data['balance'].quantile(0.75)
IQR_balance = Q3_balance - Q1_balance
threshold_balance = 1.80 * IQR_balance

# Removing outliers from 'balance'
df_no_outliers_balance = data[(data['balance'] >= Q1_balance - threshold_balance) & (data['balance'] <= Q3_balance + threshold_balance)]
data = df_no_outliers_balance
# Method for identifying outliers for the 'duration' column
Q1_duration = data['duration'].quantile(0.25)
Q3_duration = data['duration'].quantile(0.75)
IQR_duration = Q3_duration - Q1_duration
threshold_duration =1.80 * IQR_duration

# Removing outliers from 'duration'
df_no_outliers_duration = data[(data['duration'] >= Q1_duration - threshold_duration) & (data['duration'] <= Q3_duration + threshold_duration)]
data = df_no_outliers_duration
# Method for identifying outliers for the 'age' column
Q1_age = data['age'].quantile(0.25)
Q3_age = data['age'].quantile(0.75)
IQR_age = Q3_age - Q1_age
threshold_age = 1.80 * IQR_age

# Removing outliers from 'age'
df_no_outliers_age = data[(data['age'] >= Q1_age - threshold_age) & (data['age'] <= Q3_age + threshold_age)]
data = df_no_outliers_age

data.shape


# ### Now we are visualize and find the relationship of various columns with the subcription

# #### JOB vs SUBCRIPTION

# In[266]:


plt.figure(figsize=(15,10))
ax=sns.countplot(data=data, x='job', hue=data.y)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.xlabel("job")
plt.title("job vs subscription")
plt.legend(['NO', 'YES'])
plt.show()


# ### Martial status vs subscription

# In[267]:


plt.figure(figsize=(15,10))
ax=sns.countplot(data=data, x='marital', hue=data.y)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.xlabel("martial")
plt.title("Marital vs subscription")
plt.legend(['NO', 'YES'])
plt.show()


# #### By this plot we found that the divorced or widowed customers are less likely to take our subcription compared to married and single customers

# ### Education vs subcription

# In[268]:


plt.figure(figsize=(15,10))
ax=sns.countplot(data=data, x='education', hue=data.y)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.xlabel("education")
plt.title("education vs subscription")
plt.legend(['NO', 'YES'])
plt.show()


# ### Default vs subcription

# In[269]:


plt.figure(figsize=(15,10))
ax=sns.countplot(data=data, x='default', hue=data.y)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.xlabel("default")
plt.title("default vs subscription")
plt.legend(['NO', 'YES'])
plt.show()


# ### Housing loan vs subscription

# In[270]:


plt.figure(figsize=(15,10))
ax=sns.countplot(data=data, x='housing', hue=data.y)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.xlabel("housing loan status")
plt.title("housing loan vs subscription")
plt.legend(['NO', 'YES'])
plt.show()


# ### Personal loan status vs subscription

# In[271]:


plt.figure(figsize=(15,10))
ax=sns.countplot(data=data, x='loan', hue=data.y)
ax.set_xticklabels(ax.get_xticklabels(),rotation=25)
plt.xlabel("Personal loan status")
plt.title("Personal loan vs subscription")
plt.legend(['NO', 'YES'])
plt.show()


# After the visualization we found out that the majority of the customers who subcribed our terms are not married and single persons with good education and with no defauld and with no personal loans

# ## Converting the catagorical column:

# In[272]:


data['job'] = data['job'].map({
    'blue-collar': 0,
    'management': 1,
    'technician': 2,
    'admin': 3,
    'services': 4,
    'retired': 5,
    'self-employed': 6,
    'entrepreneur': 7,
    'unemployed': 8,
    'housemaid': 9,
    'student': 10,
})


data['marital'] = data['marital'].map({'single': 0 ,'married': 1 ,'divorced': 2})

data['education'] = data['education'].map({'primary': 0 ,'secondary': 1 ,'tertiary': 2})

data['default'] = data['default'].map({'no': 0 ,'yes': 1 })

data['housing'] = data['housing'].map({'no': 0 ,'yes': 1 })

data['loan'] = data['loan'].map({'no': 0 ,'yes': 1 })

data['month'] = data['month'].map({
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec':12
})

data['y'] = data['y'].map({'no': 0 ,'yes': 1 })


# Impute missing values with median (you can use other strategies as well)
data = data.fillna(data.median())



#  we are performing feature engineering and data preprocessing on the 'data' DataFrame. The goal is to transform categorical columns into numerical format by mapping their values to numerical equivalents. This process is essential for preparing the data for machine learning algorithms, as many algorithms require numerical inputs rather than categorical ones.
# 
# Here's a breakdown of what we're doing in this code:
# 
# We are mapping specific values of categorical columns to numerical values to encode them appropriately.
# Columns like 'job,' 'marital,' 'education,' 'default,' 'housing,' 'loan,' 'month,' and 'y' are being transformed to numerical representations.
# The resulting DataFrame will have these columns with numerical values, making it suitable for machine learning tasks such as classification.
# This preprocessing step is essential for building and training machine learning models, as they typically work with numerical data.

# In[273]:


data.dtypes


# In[274]:


data.head(10)


# As we can see we have converted the catagorical values into numerical values now the dataset is ready to train the model.

# # Correlation matrix

# In[275]:


corr_matrix = data.corr()
corr_matrix


# In[276]:


plt.figure(figsize=(12,10))
sns.heatmap(data.corr(),annot=True)
plt.xticks(rotation = 45, horizontalalignment = 'right', fontsize = 12)
plt.yticks(rotation = 0, fontsize = 12)
plt.show()


# ### Dividing the data into train and test data:

#  NOW LET US USE THE KFOLD OR 10FOLD STRATEGY TO TRAIN THE MODEL AND CROSS VALIDATION TO EVALUATE ITS ACCURACY , AND I AM ALSO GOING TO DO FEATURE SCALING FOR THEIR IMPROVISED PERFORMANCE
# 

# In[277]:


Y = data['y'].values
X = data.drop('y', axis=1).values

# NOW I AM SEPARATING THE TRAIN AND TEST DATA IN ORDER TO TRAIN THE MODEL
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size = 0.25, random_state=7)

models_list = []
models_list.append(('log reg', LogisticRegression()))
models_list.append(('NB', GaussianNB()))
models_list.append(('KNN', KNeighborsClassifier()))
models_list.append(('CART', DecisionTreeClassifier()))
models_list.append(('LDA',LinearDiscriminantAnalysis()))
models_list.append(('RFC',RandomForestClassifier()))

# NOW LET US USE THE KFOLD OR 10FOLD STRATEGY TO TRAIN THE MODEL AND CROSS VALIDATION TO EVALUATE ITS ACCURACY

num_folds = 10

results = []
names = []

pipelines = []

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('Scaledlog reg', Pipeline([('Scaler', StandardScaler()),('log reg', LogisticRegression( ))])))
pipelines.append(('ScaledRFC', Pipeline([('Scaler', StandardScaler()),('RFC', RandomForestClassifier())])))

print("\n\n\nAccuracies of algorithm after scaling:\n")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    kfold = KFold(n_splits=num_folds)
    for name, model in pipelines:
        start = time.time()
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        end = time.time()
        results.append(cv_results)
        names.append(name)
        print( "%s: %f (%f) (run time: %f)" % (name, cv_results.mean(), cv_results.std(), end-start))


# ### Now let us ask the model to predict the values with unknown dataset to the model.
# 

# In[231]:


scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
model = LogisticRegression( )
start = time.time()
model.fit(X_train_scaled, Y_train)   #Training of algorithm
end = time.time()
print( "\n\n LogisticRegression Training Completed. It's Run Time: %f" % (end-start))



# estimate accuracy on test dataset
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
print("All predictions done successfully by LogisticRegression Machine Learning Algorithm")
print("\n\nAccuracy score %f" % accuracy_score(Y_test, predictions))

print("\n\n")
print("confusion_matrix = \n")
print( confusion_matrix(Y_test, predictions))


report = classification_report(Y_test, predictions)
print("\n\n\nclassification_report:\n\n",report)


# ### saving the trained model

# In[234]:


import joblib
filename = "LOGISTIC REGRESSION MODEL.sav"
joblib.dump(model, filename)
print( "Model dumped successfully into a file by Joblib")


# In[236]:


scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
model = RandomForestClassifier()
start = time.time()
model.fit(X_train_scaled, Y_train)   #Training of algorithm
end = time.time()
print( "\n\n RandomForestClassifier Training Completed. It's Run Time: %f" % (end-start))



# estimate accuracy on test dataset
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)
print("All predictions done successfully by RandomForestClassifier Machine Learning Algorithm")
print("\n\nAccuracy score %f" % accuracy_score(Y_test, predictions))

print("\n\n")
print("confusion_matrix = \n")
print( confusion_matrix(Y_test, predictions))


report = classification_report(Y_test, predictions)
print("\n\n\nclassification_report:\n\n",report)


# ##  The 10 questions that need to be answered :
# 
# 
# ###  1. What is the distribution of the customer ages?

# ![image.png](attachment:image.png)

# In[237]:


customer_age = data['age']


mean_age = customer_age.mean()
median_age = customer_age.median()

percentile_25 = customer_age.quantile(0.25)
percentile_75 = customer_age.quantile(0.75)

print("Mean Age:", mean_age)
print("Median Age:", median_age)
print("25th Percentile:", percentile_25)
print("75th Percentile:", percentile_75)


# ### 2. What is the relationship between customer age and subscription?
# 

# ![image.png](attachment:image.png)

# ### I observed that when the customers age increases their probability of subscription is reducing
# ### therefore I can say that they are quite inversly proportional .

# ### 3) Are there any other factors that are correlated with subscription?

# Most of the features are releted to the subcription status . This is found by the correlation matrix

# ### 4. What is the accuracy of the logistic regression model?

# Accuracy score of logistic regression model is 0.919848
# The accuracy can further be increased but to balance the precision , f1 , recall values I found this will be optimum. 
# 

# ### 5. What are the most important features for the logistic regression model?

# In[243]:


from sklearn.linear_model import LogisticRegression


model = LogisticRegression()
model.fit(X_train, Y_train)

# Get the coefficients assigned to each feature
coefficients = model.coef_[0]

# Create a list of feature names (assuming they are in the same order as in X)
feature_names = ['age','job','marital','education','default','housing','loan','balance','contact','day','month','duration',  'campaign' , 'pdays' , 'previous', 'poutcome']

# Create a dictionary to store feature names and their corresponding coefficients
feature_coefficients = dict(zip(feature_names, coefficients))

# Sort features by absolute coefficient value (importance)
important_features = sorted(feature_coefficients.items(), key=lambda x_train: abs(x_train[1]), reverse=True)

# Display the most important features and their coefficients
for feature, coefficient in important_features:
    print(f"Feature: {feature}, Coefficient: {coefficient:.4f}")



# #### Based on the coefficient vaules 
# #### here are the top 5 features for logistic regression model
# 
# 
# 1) Loan (Coefficient: -0.8206): This is the most important feature. The negative coefficient indicates that having a loan has a significant negative impact on the likelihood of subscribing to a term deposit. Customers with loans are much less likely to subscribe.
# 
# 2) Duration (Coefficient: -0.5765): The second most important feature. A longer call duration is associated with a significantly lower likelihood of subscribing. Shorter call durations are more favorable for subscription.
# 
# 3) Balance (Coefficient: -0.2307): The balance in the customer's account is also highly influential. A higher account balance has a significant negative impact on subscription, meaning customers with higher balances are less likely to subscribe.
# 
# 4) Marital (Coefficient: -0.1270): The marital status of the customer is the fourth most important feature. Being married (or having a marital status other than single) is associated with a lower likelihood of subscribing.
# 
# 5) Pdays (Coefficient: 0.0751): The fifth most important feature. A longer time since the last contact (higher value for pdays) has a small positive impact on subscription. Customers who were contacted more recently are slightly more likely to subscribe.

# ### 6. What is the precision of the logistic regression model?

#  logistic regression model:
# 
#                precision   
# 
#            0       0.93      
#            1       0.55      

# ### 7. What is the recall of the logistic regression model?

# logistic regression model:
# 
#                  recall 
# 
#            0       0.99   
#            1       0.16   

# ### 8. What is the f1-score of the logistic regression model?

# 
# logistic regression model:
# 
#                  f1-score 
# 
#            0       0.96
#            1       0.25 

# ### 9. How can you improve the performance of the logistic regression model?

# 1) Feature Scaling:
#    feature scaling plays an important role in the accuracy of the  logistic regression model.
#   
# 2) Outlier Detection and Handling:
#     finding and removal of the outliers may potentially affect the model
#         
# 3) Cross-Validation:
#    Using k-fold cross-validation to assess the model's performance more reliably and avoid overfitting  
# 
# 
# 

# ### 10. What are the limitations of the logistic regression model?

# 1) Linearity Assumption: Logistic regression assumes a linear relationship between the features and the log-odds of the response variable. This means it may not perform well when dealing with complex, non-linear relationships in the data.
# 
# 2) Sensitivity to Outliers: Outliers in the data can significantly affect the coefficients and predictions of a logistic regression model. Proper outlier detection and handling are crucial for robust results.
# 
# 3) Limited Expressiveness: Logistic regression has limited expressiveness, which means it may struggle to capture complex interactions and patterns in the data. In cases where the underlying relationships are highly intricate, more sophisticated models might be required for better performance.

# In[ ]:




