

"""IMPORTING THE LIBRARIES

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

"""READ THE DATASET"""

df=pd.read_csv('sdss_100k_galaxy_form_burst.csv')

df.head()

"""HANDING MISSING VALUES"""

df.shape

df.info()

df.isnull().sum()

"""CHANHING THE DATATYPE OF SUBCLASS FROM OBJECT TO INT"""

df['subclass'].replace(['STARFORMING','STARBURST'],[0,1],inplace=True)
df['class'].replace(['GALAXY'],[0],inplace=True)

df.head()

df.shape

"""DESCRIPTIVE STATISTICAL"""

df.describe()

"""UNIVARIATE ANALYSIS"""

sub = df["subclass"].value_counts()
sub

sub.plot(kind="pie",subplots=True,autopct="%1.2f%%")

"""HANDLING THE OUTLIERS"""

def func(col):
  sns.boxplot(x=col,data=df)
  plt.show()
for i in df.columns:
  func(i)

# Iterate over each column
for column in df.columns:
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(df[column]):
        # Calculate quantiles
        quant = df[column].quantile(q=[0.75, 0.25])
        Q3 = quant.loc[0.75]
        Q1 = quant.loc[0.25]

        # Calculate IQR
        IQR = Q3 - Q1

        # Calculate lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Replace outliers with values within the bounds
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

# Iterate over each column and plot boxplot
for column in df.columns:
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
    sns.boxplot(x=df[column])
    plt.title(f'Boxplot for {column}')
    plt.xlabel(column)
    plt.show()

"""BIVARIATE ANALYSIS"""

sns.barplot(x='subclass',y='i',data=df)

sns.barplot(x='subclass',y='z',data=df)

"""MULTIVARIATE ANALYSIS"""

plt.figure(figsize=(30,22))
sns.heatmap(df.corr(),annot=True)
plt.show()

"""SELECTING BEST FEATURES USING SELECT K BEST"""

x=df.drop(['subclass',],axis=1)
y=df['subclass']

#i want to know top best columns in the data frame using selectkBest k=10
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
# Assuming x and y are your data and target variables
selector = SelectKBest(score_func=f_classif, k=10)  # select top 10 features
#selector = selectKBest(score_func=chi2,k=10)  # For classificaton takes with non negative features

#fit selector to the data
x_selected = selector.fit_transform(x,y) # Fixed: Use selector instead of features
# Get the names of the selected features
selected_features = x.columns[selector.get_support()] # Fixed: Use x instead of X

# print the selected features
print("selected features:",selected_features)

"""BALANCING VALUE COUNTS USING SMOTE"""

# Assuming your target column is 'subclass' in your DataFrame 'df'
x = df.drop(['subclass','class'], axis=1)
y = df['subclass']

# Initialize SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE() # Initialize the SMOTE object

x_resampled, y_resampled = smote.fit_resample(x,y)

# check the new value counts
print(pd.Series(y_resampled).value_counts())

"""SPLITTING DATA INTO TRAIN AND TEST"""

df1=df[['i','z','modelFlux_z','petroRad_g','petroRad_r','petroFlux_z','petroR50_u','petroR50_g','petroR50_i','petroR50_r','subclass']]

from sklearn.model_selection import train_test_split
x=df1[['i','z','modelFlux_z','petroRad_g','petroRad_r','petroFlux_z','petroR50_u','petroR50_g','petroR50_i','petroR50_r']]
y=df1["subclass"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

"""SCALING THE FEATURE VARIABLES USING STANDARDSCALER METHOD"""

from sklearn.preprocessing import StandardScaler
# create a scalar object
sc=StandardScaler()

# Transform your data
scaled_data = sc.fit_transform(x_train)

"""TRAINING THE MODEL IN MULTIPLE ALGORITHMS

DECISION TREE CLASSIFIER
"""

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier() # Use the correct class name with capitalization

# Train the classifier on the training data
clf.fit(x_train, y_train)

# make predictions on the testing data
y_pred = clf.predict(x_test)

# Evaluate the classifier
from sklearn.metrics import classification_report # Don't forget to import this module
report = classification_report(y_test, y_pred)
print("classification Repoprt:\n",report)

from sklearn.metrics import accuracy_score # Import the accuracy_score function
print(accuracy_score(y_pred,y_test))

print(accuracy_score(y_pred,y_test))

"""LOGISTIC REGRESSION

"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, recall_score, precision_score, confusion_matrix, f1_score

lg = LogisticRegression()
log=lg.fit(x_train,y_train)
print("confusion matrix: \n",confusion_matrix(y_test,y_pred))
print("............................................")
print("classification report:\n",classification_report(y_test,y_pred))
print("............................................")
print("accuracy score:\n",accuracy_score(y_test,y_pred))

print(accuracy_score(y_pred,y_test))

"""RANDOM FOREST CLASSIFIER"""

from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier

# Train the Random Forest classifier
RF = RandomForestClassifier()

RF.fit(x_train,y_train)
RFtrain=RF.predict(x_train)
RFtest=RF.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report # Import necessary functions

# print classification report , confusion matrix
print(confusion_matrix(RFtrain,y_train))
print(confusion_matrix(RFtest,y_test))
print(classification_report(RFtrain,y_train)) # Fix the typo here
print(classification_report(RFtest,y_test)) # Fix the typo here

print(accuracy_score(RFtrain,y_train))
print(accuracy_score(RFtest,y_test))

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # Import the accuracy_score function

print(accuracy_score(RFtrain,y_train))
print(accuracy_score(RFtest,y_test))

RF.fit(x_train,y_train) # Train the model on your training data

"""TEST THE MODEL"""

RF.predict([[16.946170,16.708910,207.218700,4.180779,4.060687,194.731000,2.141953,2.149080,2.056686,2.055798]])

RF.predict([[17.675285,17.53775,104.25655,3.397512,3.3975512,3.424717,90.717547,1.632243,1.548225,1.596137]])

"""SAVE THE MODEL"""

import pickle

pickle.dump(RF,open("RF.pkl","wb"))