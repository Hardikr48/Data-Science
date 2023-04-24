#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
pd.options.display.float_format = '{:.2f}'.format
from sklearn import preprocessing

df = pd.read_csv('final_data.csv')


# In[2]:


emp_scoreA = pd.read_csv(r"Questionnaire_datasetIA.csv", encoding= 'unicode_escape') 
emp_scoreB = pd.read_csv(r"Questionnaire_datasetIB.csv", encoding= 'unicode_escape')


# In[3]:


# Checking for the percentage of null values of each column
null_percentages = df.apply(lambda x: (x.isnull().sum() / len(df)) * 100)
print(null_percentages)


# In[4]:


# Define the list of columns to drop
cols_to_drop = ['Export date','Recording date','Recording date UTC',
                'Recording start time', 'Recording start time UTC',
                'Recording duration', 'Recording software version',
                'Recording resolution height', 'Recording resolution width',
                'Recording monitor latency', 'Recording timestamp',
                'Presented Stimulus name', 'Presented Media name',
                'Computer timestamp', 'Participant name', 'Recording name',
                'Timeline name', 'Recording Fixation filter name',
                'Unnamed: 0', 'Event', 'Event value',
                'Pupil diameter left', 'Pupil diameter right',
                'Fixation point X', 'Fixation point Y',
                'Fixation point X (MCSnorm)', 'Fixation point Y (MCSnorm)',
                'Mouse position X', 'Mouse position Y',
                'Eyetracker timestamp', 'Presented Media height',
                'Presented Media position Y (DACSpx)']

# Drop the specified columns
df = df.drop(columns=cols_to_drop)


# In[5]:


cols = ['Validity left', 'Validity right', 'Sensor']
for col in cols:
    df[col].fillna('Not Recorded', inplace=True)


# In[6]:


df.drop('Unnamed: 0.1',axis = 1,inplace = True)


# In[7]:


df.head()


# In[8]:


df['Eye movement type'].unique()


# In[9]:


#Replacing the , to . for these columns and convert their type from object to float
string_replace = ['Eye position left X (DACSmm)',
 'Eye position left Y (DACSmm)',
 'Eye position left Z (DACSmm)',
 'Eye position right X (DACSmm)',
 'Eye position right Y (DACSmm)',
 'Eye position right Z (DACSmm)',
 'Gaze point left X (DACSmm)',
 'Gaze point left Y (DACSmm)',
 'Gaze point right X (DACSmm)',
 'Gaze point right Y (DACSmm)',
 'Gaze point X (MCSnorm)',
 'Gaze point Y (MCSnorm)',
 'Gaze point left X (MCSnorm)',
 'Gaze point left Y (MCSnorm)',
 'Gaze point right X (MCSnorm)',
 'Gaze point right Y (MCSnorm)',
 'Gaze direction left X',
 'Gaze direction left Y',
 'Gaze direction left Z',
 'Gaze direction right X',
 'Gaze direction right Y',
 'Gaze direction right Z']

# Convert the type of the column from object to float
df[string_replace] = df[string_replace].astype(str).replace(',', '.', regex=True).astype(float)


# In[17]:


import gc
from sklearn import preprocessing

def clean_data(df):
    # Clear memory
    gc.collect()
    
    # Mean Columns
    mean_columns = ['Gaze point X',
     'Gaze point Y',
     'Gaze point left X',
     'Gaze point left Y',
     'Gaze point right X',
     'Gaze point right Y',
     'Gaze direction left X',
     'Gaze direction left Y',
     'Gaze direction left Z',
     'Gaze direction right X',
     'Gaze direction right Y',
     'Gaze direction right Z',
     'Eye position left X (DACSmm)',
     'Eye position left Y (DACSmm)',
     'Eye position left Z (DACSmm)',
     'Eye position right X (DACSmm)',
     'Eye position right Y (DACSmm)',
     'Eye position right Z (DACSmm)',
     'Gaze point left X (DACSmm)',
     'Gaze point left Y (DACSmm)',
     'Gaze point right X (DACSmm)',
     'Gaze point right Y (DACSmm)',
     'Gaze point X (MCSnorm)',
     'Gaze point Y (MCSnorm)',
     'Gaze point left X (MCSnorm)',
     'Gaze point left Y (MCSnorm)',
     'Gaze point right X (MCSnorm)',
     'Gaze point right Y (MCSnorm)',
     'Gaze direction left X',
     'Gaze direction left Y',
     'Gaze direction left Z',
     'Gaze direction right X',
     'Gaze direction right Y',
     'Gaze direction right Z']

    # Fill NaN values in each column with the mean of those column
    for col in df[mean_columns]:
        mean_val = df[col].mean()
        df[col].fillna(mean_val, inplace=True)
        print("Done",col)

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Remove Test group experiment and Control group experiment from the final dataframe
    final_df = df[~(df['Project name'] == 'Test group experiment') & ~(df['Project name'] == 'Control group experiment')]

    # Convert columns to Label Encoders
    label_converter = ['Sensor', 'Validity left', 'Validity right', 'Eye movement type', 'Project name']
    le = preprocessing.LabelEncoder()
    for i in label_converter:
        final_df[i] = le.fit_transform(final_df[i])

    final_df.reset_index(drop=True, inplace=True)
    
    return final_df


# In[21]:


final_df = clean_data(df)


# In[22]:


df.shape, final_df.shape


# In[23]:


final_df.head()


# In[24]:


final_df.corr().transpose()


# In[26]:


def plot_corr_heatmap(final_df):
    corr_matrix = final_df.corr().transpose()
    plt.figure(figsize=(20, 10))
    sns.heatmap(corr_matrix, cmap='magma')
    plt.title('Correlation Heatmap')
    plt.show()
plot_corr_heatmap(final_df)


# In[27]:


final_df.duplicated().sum()


# In[28]:


final_df.drop_duplicates(inplace = True)


# In[29]:


final_df.isnull().sum().sum()


# In[30]:


final_df['Sensor'].unique()


# In[31]:


final_df.drop('Sensor',axis = 1,inplace = True)


# In[32]:


final_df['Project name'].unique()


# In[33]:


#seperating dependent and independent variable x,y
X = final_df.drop('Eye movement type',axis = 1)
y = final_df['Eye movement type']


# In[34]:


from sklearn.model_selection import train_test_split

#splitting the data into training 67 % and testing 33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[35]:


X_train.shape,y_train.shape


# In[36]:


X_test.shape,y_test.shape


# In[37]:


gc.collect()


# In[38]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

classify_1 = GaussianNB()
classify_1.fit(X_train,y_train)


# In[39]:


pred1 = classify_1.predict(X_test)


# In[40]:


print(classification_report(y_test, pred1))


# In[41]:


# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

c1 = LogisticRegression()
c1.fit(X_train,y_train)


# In[42]:


pred2 = c1.predict(X_test)


# In[43]:


print(classification_report(y_test, pred2))


# In[44]:


# Empathy Score


# In[48]:


def get_final_empathy_scores(emp_scoreA, emp_scoreB):
    final_empathy_scoreA_columns = []
    for i in emp_scoreA.columns:
        if emp_scoreA[i].dtypes != 'O':
            final_empathy_scoreA_columns.append(i)

    final_empathy_scoreA = emp_scoreA[final_empathy_scoreA_columns]
    final_empathy_scoreB = emp_scoreB[final_empathy_scoreA_columns]
    
    return final_empathy_scoreA, final_empathy_scoreB

final_empathy_scoreA, final_empathy_scoreB = get_final_empathy_scores(emp_scoreA, emp_scoreB)


# In[49]:


final_empathy_scoreA.drop('NR',axis = 1,inplace = True)
final_empathy_scoreB.drop('NR',axis = 1,inplace = True)


# In[50]:


final_empathy_scoreA.describe()


# In[51]:


final_empathy_scoreA.corr()


# In[52]:


plot_corr_heatmap(final_empathy_scoreA)


# In[53]:


plot_corr_heatmap(final_empathy_scoreB)


# In[54]:


def prepare_empathy_training_data(final_empathy_scoreA):
    X1_train = final_empathy_scoreA.drop('Total Score extended', axis=1)
    y1_train = final_empathy_scoreA['Total Score extended']
    return X1_train, y1_train

X1_train, y1_train = prepare_empathy_training_data(final_empathy_scoreA)


# In[55]:


def prepare_empathyB_data():
    #making EmpathyB csv for testing the empathy score
    X1_test = final_empathy_scoreB.drop('Total Score extended',axis = 1)
    y1_test = final_empathy_scoreB['Total Score extended']
    return X1_test, y1_test

X1_test, y1_test = prepare_empathyB_data()


# In[56]:


# Linear Regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math

def linear_regression(X_train, y_train, X_test, y_test):
    predict_1 = LinearRegression()
    predict_1.fit(X_train, y_train)
    score_predicted1 = predict_1.predict(X_test)
    
    mse = mean_squared_error(y_test, score_predicted1)
    r2 = r2_score(y_test, score_predicted1)
    rmse = math.sqrt(mse)
    
    print('Root mean square of LR for predicting empathy score is:', rmse)
    print('R2 of LR for predicting empathy score is:', r2)
    
    return rmse, r2

rmse, r2 = linear_regression(X1_train, y1_train, X1_test, y1_test)


# In[58]:


# Random Forest
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score

def random_forest_regression(X_train, y_train, X_test, y_test):
    predict_2 = RandomForestClassifier()
    predict_2.fit(X_train, y_train)
    score_predicted2 = predict_2.predict(X_test)
    mse_1 = mean_squared_error(y_test, score_predicted2)
    r2_1 = r2_score(y_test, score_predicted2)
    print('Root mean square of RF for predicting empathy score is:- ', math.sqrt(mse_1))
    print('R2 of RF for predicting empathy score is:- ', r2_1)


# In[59]:


random_forest_regression(X1_train, y1_train, X1_test, y1_test)


# In[ ]:




