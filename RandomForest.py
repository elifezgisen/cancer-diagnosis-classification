import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import datasets

#Importing data:
df = pd.read_csv('data.csv')

# Shape of the dataset. (Rows and Columns)
df.shape

df.head(360)

# Return statistical values of numeric data columns.
df.describe()

# Checking for null values in columns. (NAN, NaN, na)
df.isna().sum()


# How many different custom results are in the "disease_type" column, which is the actual target column in the dataset.

# Colon Cancer - Lung Cancer - Breast Cancer - Prosrtate Cancer = 4 Class

df.disease_type.unique()

#How many of these particular results are found in the "disease_type" column.
df['disease_type'].value_counts()

# Visualization of the "disease_type" column.
sns.countplot(x='disease_type', data=df, palette='husl')
plt.ylabel=('Count')
plt.xlabel('Disease Type')

plt.title('Distribution of Disease Types')
plt.show()

list(df.columns)

# Checking data types to see if there is a situation that needs coding.
df.dtypes

# Model Training:

# X = Feature - Independent
# Y = Label/Target - Dependant

# Separating the target column from the features.

X = df.drop("disease_type", axis = 1).values #Features - Independent (All columns except the target column)
Y = df["disease_type"].values #Target - Dependent

type(X)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0) 

sc = StandardScaler() 
X_train = sc.fit_transform(X_train) 
X_test = sc.fit_transform(X_test)


forest = RandomForestClassifier(n_estimators = 50)
model1 = forest.fit(X_train, Y_train)

print(forest.score(X_train, Y_train))


# Confusion Matrix:

prediction = model1.predict(X_test)

cm1 = confusion_matrix(Y_test, prediction)
print(cm1)


# Model's Metrics - Classification Report and Accuracy:

print(classification_report(Y_test, prediction))

print('Accuracy: ', accuracy_score(Y_test, prediction))

sensitivity = cm1[0,0]/(cm1[0,0]+cm1[0,1])
print('Sensitivity : ', sensitivity )

specificity = cm1[1,1]/(cm1[1,0]+cm1[1,1])
print('Specificity : ', specificity)

BreastNumber = cm1[0, 0] + cm1[0, 1] + cm1[0, 2] + cm1[0, 3]
ColonNumber = cm1[1, 0] + cm1[1, 1] + cm1[1, 2] + cm1[1, 3]
LungNumber = cm1[2, 0] + cm1[2, 1] + cm1[2, 2] + cm1[2, 3]
ProstateNumber = cm1[3, 0] + cm1[3, 1] + cm1[3, 2] + cm1[3, 3]


# Breast Cancer
print('Breast Cancer:')
print('-------------------')

BreastSensitivity = cm1[0, 0 ] / BreastNumber
print('BreastSensitivity : ', BreastSensitivity)

BreastSpecificity = (cm1[1, 1] + cm1[2, 2] + cm1[3, 3]) / (ColonNumber + LungNumber + ProstateNumber)
print('BreastSpecificity : ', BreastSpecificity)


print('\n')
# Colon Cancer
print('Colon Cancer:')
print('------------------')

ColonSensitivity = cm1[1,1] / ColonNumber
print('ColonSensitivity : ', ColonSensitivity)

ColonSpecificity = (cm1[0, 0] + cm1[2, 2] + cm1[3, 3]) / (BreastNumber + LungNumber + ProstateNumber)
print('ColonSpecificity : ', ColonSpecificity)


print('\n')
# Lung Cancer
print('Lung Cancer:')
print('-----------------')

LungSensitivity = cm1[2,2] / LungNumber
print('LungSensitivity : ', LungSensitivity)

LungSpecificity = (cm1[0, 0] + cm1[1,1] + cm1[3, 3]) / (BreastNumber + ColonNumber + ProstateNumber)
print('LungSpecificity : ', LungSpecificity)

print('\n')
# Prostate Cancer
print('Prostate Cancer:')
print('---------------------')

ProstateSensitivity = cm1[3,3] / ProstateNumber
print('ProstateSensitivity : ', ProstateSensitivity)

ProstateSpecificity = (cm1[0, 0] + cm1[1,1] + cm1[2,2]) / (BreastNumber + ColonNumber +LungNumber)
print('ProstateSpecificity : ', ProstateSpecificity)

