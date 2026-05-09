import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Data Collection & Processing"""

# load the data from csv file to Pandas DataFrame
titanic_data = pd.read_csv(r'C:\Users\bhagy\OneDrive\Desktop\DIPLOMA STUDY MATERIAL\Diploma sem 5 study material\IML microproject\train.csv')


# printing the first 5 rows of the dataframe
titanic_data.head()

# number of rows and Columns
titanic_data.shape

# getting some informations about the data
titanic_data.info()

# check the number of missing values in each column
titanic_data.isnull().sum()

"""Handling the Missing values"""

# drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# replacing the missing values in "Age" column with mean value
titanic_data['Age']=titanic_data['Age'].fillna(titanic_data['Age'].mean())

# finding the mode value of "Embarked" column
print(titanic_data['Embarked'].mode())

print(titanic_data['Embarked'].mode()[0])

# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked']=titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0])

# check the number of missing values in each column
titanic_data.isnull().sum()

"""Data Analysis"""

# getting some statistical measures about the data
titanic_data.describe()

# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()

"""Data Visualization"""

sns.set()

# making a count plot for "Survived" column
sns.countplot(x='Survived', data=titanic_data)

titanic_data['Sex'].value_counts()

# making a count plot for "Sex" column
sns.countplot(x='Sex', data=titanic_data)

# number of survivors Gender wise
sns.countplot(x='Sex', hue='Survived', data=titanic_data)

# making a count plot for "Pclass" column
sns.countplot(x='Pclass', data=titanic_data)

sns.countplot(x='Pclass', hue='Survived', data=titanic_data)

"""Encoding the Categorical Columns"""

titanic_data['Sex'].value_counts()

titanic_data['Embarked'].value_counts()


# Set the option to enable future behavior agad edit nai thaay
pd.set_option('future.no_silent_downcasting', True)

# Perform the replacement change karse value
titanic_data.replace({'Sex': {'male': 0, 'female': 1}, 
                      'Embarked': {'S': 0, 'C': 1, 'Q': 2}}, 
                     inplace=True)



titanic_data.head()

"""Separating features & Target"""

X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']

print(X)

print(Y)

"""Splitting the data into training data & Test data"""

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

"""Model Training

Logistic Regression
"""

model = LogisticRegression()

# training the Logistic Regression model with training data
model.fit(X_train, Y_train)

"""Model Evaluation

Accuracy Score
"""

# accuracy on training data
X_train_prediction = model.predict(X_train)

print(X_train_prediction)

training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)

print(X_test_prediction)

test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)

# Print classification report
from sklearn.metrics import classification_report
print(classification_report(Y_test, X_test_prediction))


## Create a count plot for survival
#sns.countplot(x='Survived', data=titanic_data)
#plt.title('Count of Survivors on the Titanic')
#plt.xlabel('Survived (0 = No, 1 = Yes)')
#plt.ylabel('Count')
#plt.show()  # This will display the graph
#aane kaam ma nathi levanu but back up maate rakhyu che

import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Set the visual style of the plots
sns.set(style='whitegrid')

# 1. Count plot for survival
plt.figure(figsize=(10, 5))
sns.countplot(x='Survived', data=titanic_data, palette='pastel')
plt.title('Count of Survivors on the Titanic', fontsize=16)
plt.xlabel('Survived (0 = No, 1 = Yes)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=0)
plt.show()

# 2. Count plot for survival by gender
plt.figure(figsize=(10, 5))
sns.countplot(x='Sex', hue='Survived', data=titanic_data, palette='pastel')
plt.title('Survival Count by Gender', fontsize=16)
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=0)
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# 3. Count plot for survival by passenger class
plt.figure(figsize=(10, 5))
sns.countplot(x='Pclass', hue='Survived', data=titanic_data, palette='pastel')
plt.title('Survival Count by Passenger Class', fontsize=16)
plt.xlabel('Passenger Class', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=0)
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# 4. Confusion Matrix for the Model's Predictions
confusion = confusion_matrix(Y_test, X_test_prediction)
cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=['Not Survived', 'Survived'])
plt.figure(figsize=(8, 6))
cm_display.plot(cmap='Blues', ax=plt.gca())
plt.title('Confusion Matrix', fontsize=16)
plt.show()


# Optional: Print the classification report to the console
from sklearn.metrics import classification_report
print(classification_report(Y_test, X_test_prediction))


