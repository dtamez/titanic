#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2016 Danny Tamez <zematynnad@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Kaggle getting started machine learning problem dealing with trying to predict
likelihood that someone on the Titanic would die.
"""

import numpy as np

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

train = pd.read_csv('train.csv', dtype={"Age": np.float64},)
test = pd.read_csv('test.csv', dtype={"Age": np.float64},)


def fix_nulls(dataset):
    # deal with blanks
    dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].mean())
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    return dataset


def convert_text_values(dataset):
    # deal with text values
    dataset.loc[dataset['Sex'] == 'male', 'Sex'] = 1
    dataset.loc[dataset['Sex'] == 'female', 'Sex'] = 0
    dataset.loc[dataset['Embarked'] == 'S', 'Embarked'] = 0
    dataset.loc[dataset['Embarked'] == 'C', 'Embarked'] = 1
    dataset.loc[dataset['Embarked'] == 'Q', 'Embarked'] = 2

    return dataset

data_train = fix_nulls(train)
data_train = convert_text_values(data_train)
data_test = fix_nulls(test)
data_test = convert_text_values(data_test)

columns = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Embarked']

X_train = data_train[columns].values
y_train = data_train['Survived'].values
X_test = data_test[columns].values

select = SelectKBest(k='all')
clf = LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)
#  clf = RandomForestClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

steps = [('feature_selection', select),
         ('reduce_dim', PCA()),
         ('rbm', rbm),
         ('clf', clf)]
parameters = dict(feature_selection__k=[5, 6],
                  clf__C=[1, 100, 1000],
                  rbm__learning_rate=[0.25, 0.5, 0.75, 1.0],
                  rbm__n_iter=[20],
                  rbm__n_components=[50, 100, 150]
                  )


pipeline = Pipeline(steps)
cv = GridSearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
predictions = cv.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})
submission.to_csv('predictions.csv', index=False)
