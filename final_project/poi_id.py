#!/usr/bin/python

# Info on the challenge:
# Mark to market accounting, mark the value of a future project and count it as income
# Documentary to watch:
# The Smartest Guys in the Room

import sys
sys.path.append("../tools/")

import pickle
import pandas as pd
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from Custom import ScorePredictions
from tester import dump_classifier_and_data, test_classifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif


'''
Pre-processed Code Setup
'''

poi_feature = ['poi']

finance_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                    'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
                    'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                    'restricted_stock', 'director_fees']

# Removing email_address to focus only numeric features
email_features = ['to_messages', 'from_poi_to_this_person',
                  'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

# Task 1: Select features

features_list = poi_feature + finance_features + email_features

with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

#find total amount of targets of interest
interest = 0
for x in data_dict:
    print(data_dict[x])
    if data_dict[x]['poi'] == True:
        interest += 1

print('Target of Interest:', interest)

# Task 2: Remove outliers

# Outliers, specifically on the high end are likely to contain many PoI's.  For outlier we will remove
# erroneous records as well as unusual or null values.

data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)

my_frame = pd.DataFrame.from_dict(data_dict, orient='index')

print("Data Frame Shape:", my_frame.shape)
print("==================================================================")
print("Brief look at Data Set")
print("==================================================================")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(my_frame)

my_frame = my_frame.replace('NaN',0)


# Task 3: Create new feature(s)

bps = 'Bonus_Percentage_Salary'
pois = my_frame["poi"].data

my_frame[bps] = my_frame['bonus'] / my_frame['salary']
features_list.append(bps)

my_frame = my_frame.replace([np.inf, -np.inf, np.nan], 'NaN').replace('NaN',0)

# Implement Feature Scaling for better Estimator performance.

numeric_frame = my_frame\
    .drop("email_address", axis=1)\
    .drop("poi", axis=1)

scaler = StandardScaler()

X = scaler.fit_transform(numeric_frame)

trained_frame = pd.DataFrame(X, columns=numeric_frame.columns)
trained_frame["poi"] = pois

'''
Post-processed Project Setup
'''

result_frame = trained_frame

print("==================================================================")
print("Brief look at prepared Data Set")
print("==================================================================")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(result_frame)

data_dict = result_frame.to_dict(orient="index")
my_dataset = data_dict

# Task 4: Try a varity of classifiers

data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(random_state=62, test_size=.3, n_splits=10)

features_train = []
features_test = []
labels_train = []
labels_test = []
for train_idx, test_idx in cv.split(features, labels):
    for ii in train_idx:
        features_train.append(features[ii])
        labels_train.append(labels[ii])
    for jj in test_idx:
        features_test.append(features[jj])
        labels_test.append(labels[jj])

gauss = GaussianNB()
gauss.fit(features_train, labels_train)
g_pred = gauss.predict(features_test)
ScorePredictions("Naive Bayes", labels_test, g_pred)

svm = SVC(kernel="poly")
svm.fit(features_train, labels_train)
s_pred = svm.predict(features_test)
ScorePredictions("Support Vector Machine", labels_test, s_pred)

rando_forest = RandomForestClassifier(max_leaf_nodes=2, max_depth=2)
rando_forest.fit(features_train, labels_train)
r_pred = rando_forest.predict(features_test)
ScorePredictions("Random Forest", labels_test, r_pred)

ada_boost = AdaBoostClassifier(DecisionTreeClassifier())
ada_boost.fit(features_train, labels_train)
a_pred = ada_boost.predict(features_test)
ScorePredictions("AdaBoost", labels_test, a_pred)

# Task 5: Tune your classifier

# select the best performing features...  Normally this would pipeline; however, to work with tester class
# I am doing it manually.
kbest = SelectKBest(f_classif, k=12)
kbest.fit(features_train, labels_train)
kbest_features = kbest.get_support()
kbest_scores = kbest.scores_
new_features = []
scores = []

# Zip into a new feature list
for bool, feature, score in zip(kbest_features, features_list, kbest_scores):
    if bool:
        new_features.append(feature)
        scores.append(feature + " " + str(score))

print(scores)

# Replace old features_list for tester class call compatibility
features_list = new_features

# Perform a second split with new features.
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(random_state=72, test_size=.3, n_splits=10)

features_train = []
features_test = []
labels_train = []
labels_test = []
for train_idx, test_idx in cv.split(features, labels):
    for ii in train_idx:
        features_train.append(features[ii])
        labels_train.append(labels[ii])
    for jj in test_idx:
        features_test.append(features[jj])
        labels_test.append(labels[jj])
'''
clf = RandomForestClassifier(criterion='gini', max_depth=8, max_features=None
                             n_estimators=50, n_jobs=-1, class_weight='balanced')

clf = RandomForestClassifier(criterion='entropy', max_depth=4, max_features="sqrt",
                             n_estimators=40, n_jobs=-1, class_weight='balanced')
'''
clf = RandomForestClassifier(criterion='entropy', max_depth=2, max_features="log2",
                             n_estimators=30, n_jobs=-1, class_weight='balanced')


clf.fit(features_train, labels_train)

tune_pred = clf.predict(features_test)

# Mostly used as a sanity check to make sure we don't hit any exceptions.
ScorePredictions("Tuned Random Forest", labels_test, tune_pred)

test_classifier(clf, my_dataset, features_list)

alt_features_list = features_list + [bps]

print("Testing with added feature")
test_classifier(clf, my_dataset, alt_features_list)

# Task 6: Export data

dump_classifier_and_data(clf, my_dataset, features_list)
