import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

train = pd.read_csv('train.csv')
print(train.head())

print(train.info())

# how to visualize this in PyCharm
#sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
#plt.show()

sns.set_style('whitegrid')
#sns.countplot(x='Survived', hue='Pclass', data=train)
#plt.show()

# age distribution
#sns.distplot(train['Age'].dropna(),kde=False,bins=30)
#plt.show()

# filling in missing data
#sns.boxplot(x='Pclass', y='Age', data=train)
#plt.show()


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()
# w kolumnie Age brak wartości null

# usuwanie kolumny Kabina
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)

# converting categorical features to dummy variables
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train,sex, embark],axis=1)

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.drop('PassengerId', axis=1, inplace=True)

print(train.head())

X = train.drop('Survived', axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

