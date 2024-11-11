###################################################################################
# Machine Learning Assignment 1 - Group Classification Algorithms and cleaning data
# James McGrath - November 2024
###################################################################################

import pandas as pd
import numpy as np
import seaborn as sns # Seaborn is a visualization library based on matplotlib
import matplotlib.pyplot as plt
import csv
from collections import Counter
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

verbose = 1
save_intermediate_files = 1

use_under_sampling_before_smote = 0
use_smote = 1
use_under_sampling_after_smote = 0

###################################################################################
# Data read, initial cleanup and merge for Peugeot journeys
###################################################################################

# Read the CSV files in to data frames
peugeot_01 = pd.read_csv("peugeot_207_01.csv", sep=";")

if verbose == 1:
    print("\n\nPrinting Peugeot Journey 1 Info")
    print( peugeot_01.info() )

peugeot_02 = pd.read_csv("peugeot_207_02.csv", sep=";")
if verbose == 1:
    print("\n\nPrinting Peugeot Journey 2 Info")
    print( peugeot_02.info() )

# Merge the DataFrames
peugeot = pd.concat([peugeot_01, peugeot_02])

# Insert a new column brand to distinguish each car model when merge with Opel
peugeot["brand"] = "peugeot"

if verbose == 1:
    print("\n\nPrinting Peugeot Merged Journey Info")
    print( peugeot.info() )

print('Printing the null values in peugeot data frame')
print( peugeot.isnull().sum() )

'''
This shows there are null values in some rows, but its a low number:.
The following columns have rows with null entries.
VehicleSpeedInstantaneous    9
EngineLoad                   5
EngineCoolantTemperature     5
ManifoldAbsolutePressure     5
EngineRPM                    5
MassAirFlow                  5
IntakeAirTemperature         5
FuelConsumptionAverage       5
'''
# Just drop the null values
peugeot.dropna(inplace=True)

print('\n\nPrinting the sum of null values in peugeot data frame')
print( peugeot.isnull().sum() )
print('\n\nPrinting the sum of na values in peugeot data frame')
print( peugeot.isna().sum() )

if verbose == 1:
    print("\n\nPrinting the cleaned up peugeot data frame")
    print( peugeot.info() )
    peugeot.to_csv('IntermediateDataPlots/peugeot.csv', index=False)

###################################################################################
# Data read, initial cleanup and merge for Opel journeys
###################################################################################

opel_01 = pd.read_csv("opel_corsa_01.csv", sep=";")

if verbose == 1:
    print("\n\nPrinting Opel Journey 1 Info")
    print( opel_01.info() )

opel_02 = pd.read_csv("opel_corsa_02.csv", sep=";")

if verbose == 1:
    print("\n\nPrinting Opel Journey 2 Info")
    print( opel_02.info() )

# Merge the DataFrames
opel = pd.concat([opel_01, opel_02])

# Insert a new column brand to distinguish from Peugeot
opel["brand"] = "opel"

if verbose == 1:
    print("\n\nPrinting Opel Merged Journey Info")
    print( opel.info() )
    opel.to_csv('IntermediateDataPlots/opel.csv', index=False)

print('\n\nPrinting the sum of null values in opel data frame')
print( opel.isnull().sum() )
print('\n\nPrinting the sum of na values in opel data frame')
print( opel.isna().sum() )

# No null or na values in the opel data frame

###################################################################################
# Merge Opel and Peugeot data and drop some unnecessary data
###################################################################################

# Concatenate both opel and peugeot
merged_dataset = pd.concat([peugeot, opel])

if verbose == 1:
    print("\n\nPrinting Opel and Peugeot Merged Info")
    print( merged_dataset.info() )
    sliced_df = merged_dataset.head(5)
    sliced_df = pd.concat([sliced_df, merged_dataset.tail(5)], ignore_index=True)
    print (sliced_df)
    if save_intermediate_files == 1:
        merged_dataset.to_csv('IntermediateDataPlots/merged.csv', index=False)

# We only care about predicting driving style here, so drop the roadSurface and
# traffic columns, can also drop Unnamed
trimmed_dataset = merged_dataset.drop(columns=['Unnamed: 0', 'roadSurface', 'traffic', 'brand'])
if verbose == 1:
    print("\n\nPrinting Trimmed Merged Dataset Info")
    print( trimmed_dataset.info() )

###################################################################################
# Feature Selection
###################################################################################

# Determine if a feature is useful
numeric_dataset = trimmed_dataset.select_dtypes(include=[np.number])

# Calculate the correlation matrix for numeric columns
corr_matrix = numeric_dataset.corr()

# Plot the heatmap
ax = sns.heatmap(corr_matrix, annot=True, cmap="RdYlGn", annot_kws={"size":10})
ax.figure.subplots_adjust(bottom = 0.25)
plt.show()

'''
Will want to return to this, there are a number of features which are highly 
correlated so can experiment with dropping some of them.
'''

#feature_selected_dataset = trimmed_dataset.drop(columns=['MassAirFlow', 'VehicleSpeedInstantaneous'])
feature_selected_dataset = trimmed_dataset

###################################################################################
# Label encoding driving style and brand
###################################################################################

# Need to encode the drivingSyle into a numeric value

driving_style = feature_selected_dataset['drivingStyle']
X = feature_selected_dataset.drop(columns=['drivingStyle'])

encoder = LabelEncoder()
encoded_ds = encoder.fit_transform(driving_style)
# Convert the array to a DataFrame
y = pd.DataFrame(encoded_ds, columns=['encodedDrivingStyle'])
if verbose == 1:
    print("\n\nLabel encoded dataset")
    print( y.info() )

if save_intermediate_files == 1:
    y.to_csv('IntermediateDataPlots/y.csv', index=False)
    X.to_csv('IntermediateDataPlots/X.csv', index=False)

if verbose == 1:
    print("\n\nDataset after encoding drivingStyle")
    print( X.info() )

###################################################################################
# Feature scaling and splitting into training/testing
###################################################################################

# Feature scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split the data between testing and training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=68)

if save_intermediate_files == 1:
    X_train_df = pd.DataFrame(X_train)
    X_train_df.to_csv('IntermediateDataPlots/X_train.csv')

###################################################################################
# Addressing class imbalance
###################################################################################

if verbose == 1:
    value_driving_peugeot = peugeot['drivingStyle'].value_counts()
    print('\n\nPrinting the count of driving style values from the Peugeot dataset')
    print(value_driving_peugeot)

    value_driving_opel = opel['drivingStyle'].value_counts()
    print('\n\nPrinting the count of driving style values from the Opel dataset')
    print(value_driving_opel)

    value_driving_y = y.value_counts()
    print('\n\nPrinting the count of driving style values from the merged dataset')
    print(value_driving_y)
    
    n_agressive_samples = sum(y_train.encodedDrivingStyle == 0)
    n_evenpace_samples = sum(y_train.encodedDrivingStyle == 1)
    print( "\n\nNumber of Agressive driving style samples: ", n_agressive_samples )
    print( "Number of EvenPace driving style samples: ", n_evenpace_samples )

'''
Peugeot: EvenPaceStyle 11962, AggressiveStyle 670. Minority class outnumbered almost 18:1
Opel: EvenPaceStyle: 9041, AggressiveStyle: 2089. Minority class outnumbered over 4:1
Merged: EvenPaceStyle: 21003 AgressiveStyle: 2759 Minority class outnumbered over 7:1
There is no formal definiion of imbalance, but I suspect we do have an imbalance here,
at least with the Peugeot data.
'''

# Undersampling Before
X_train_A = None
y_train_A = None
if use_under_sampling_before_smote == 1:
    undersample = RandomUnderSampler(sampling_strategy=0.5)
    X_train_A, y_train_A = undersample.fit_resample(X_train, y_train)
    if verbose == 1:
        print("\n\nShape of X_train_under (before smote): ", np.shape(X_train_A) )
        print( "Shape of y_train_under (before smote): ", np.shape(y_train_A) )
        n_agressive_samples_under = sum(y_train_A.encodedDrivingStyle == 0)
        n_evenpace_samples_under = sum(y_train_A.encodedDrivingStyle == 1)
        print( "\n\nNumber of Agressive driving style samples undersampled (before smote): ", n_agressive_samples_under )
        print( "Number of EvenPace driving style samples undersampled (before smote): ", n_evenpace_samples_under )
else:
    X_train_A = X_train
    y_train_A = y_train

# SMOTE
X_train_B = None
y_train_B = None
if use_smote == 1:
    smote = SMOTE(random_state=68, sampling_strategy=1.0)
    X_train_B, y_train_B = smote.fit_resample( X_train_A, y_train_A )
    if verbose == 1:
        print("\n\nShape of X_train_smote: ", np.shape(X_train_B) )
        print( "Shape of y_train_smote: ", np.shape(y_train_B) )
        n_agressive_samples_smote = sum(y_train_B.encodedDrivingStyle == 0)
        n_evenpace_samples_smote = sum(y_train_B.encodedDrivingStyle == 1)
        print( "\n\nNumber of Agressive driving style samples smote: ", n_agressive_samples_smote )
        print( "Number of EvenPace driving style samples smote: ", n_evenpace_samples_smote )
else:
    X_train_B = X_train_A
    y_train_B = y_train_A

# Undersampling After
X_train_final = None
y_train_final = None
if use_under_sampling_after_smote == 1:
    undersample = RandomUnderSampler(sampling_strategy=1.0)
    X_train_final, y_train_final = undersample.fit_resample(X_train_B, y_train_B)
    if verbose == 1:
        print("\n\nShape of X_train_under  (after smote): ", np.shape(X_train_final) )
        print( "Shape of y_train_under  (after smote): ", np.shape(y_train_final) )
        n_agressive_samples_under = sum(y_train_final.encodedDrivingStyle == 0)
        n_evenpace_samples_under = sum(y_train_final.encodedDrivingStyle == 1)
        print( "\n\nNumber of Agressive driving style samples undersampled (after smote): ", n_agressive_samples_under )
        print( "Number of EvenPace driving style samples undersampled (after smote): ", n_evenpace_samples_under )

    if save_intermediate_files == 1:
        X_train_under_df = pd.DataFrame(X_train_final)
        X_train_under_df.to_csv('IntermediateDataPlots/X_train_under_after_smote.csv')
        y_train_under_df = pd.DataFrame(y_train_final)
        y_train_under_df.to_csv('IntermediateDataPlots/y_train_under_after_smote.csv')
else:
    X_train_final = X_train_B
    y_train_final = y_train_B

###################################################################################
# Logistic Regression
###################################################################################

print( "\n\n--- Logistic Regression ---\n\n" )

param_grid_lgr = {'C': [0.0001, 0.005,0.001, 0.005, 0.01,0.1,1,10,100,1000]}

lgr = LogisticRegression()

lgr_gscv = GridSearchCV(lgr, param_grid_lgr, cv=5)

lgr_gscv.fit(X_train_final, np.ravel(y_train_final))

# Print the best parameters and the best score
print("LGR Best Parameters: ", lgr_gscv.best_params_)
print("LGR Best Score: ", lgr_gscv.best_score_)

lgr_clf = lgr_gscv.best_estimator_
y_pred_lgr = lgr_clf.predict(X_test)

print( "\n\nPRINT RESULTS FOR Logistic Regression" )
print("Accuracy:", accuracy_score(y_test, y_pred_lgr))
print(classification_report(y_test, y_pred_lgr))

print( confusion_matrix( y_test, y_pred_lgr ) )

###################################################################################
# K Nearest neighbors
###################################################################################

print( "\n\n--- k Nearest Neighbours ---\n\n" )

param_grid_knn = {'n_neighbors': range(1, 31), 'weights': ['uniform', 'distance']}

knn = KNeighborsClassifier()

# Configure grid search with cross-validation to evaluate all possible combinations of parameters in the grid.
knn_gscv = GridSearchCV(knn, param_grid_knn, cv=5)

knn_gscv.fit(X_train_final, np.ravel(y_train_final))

# Print the best parameters and the best score
print("KNN Best Parameters: ", knn_gscv.best_params_)
print("KNN Best Score: ", knn_gscv.best_score_)

# Use the best hyperparameters found to fit the kNN model and predict the test set outcomes.
knn_clf = knn_gscv.best_estimator_
y_pred_knn = knn_clf.predict(X_test)

print( "\n\nPRINT RESULTS FOR KNN" )
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

print( confusion_matrix( y_test, y_pred_knn ) )


###################################################################################
# ROC and AUC
###################################################################################

probs_lgr = lgr_clf.predict_proba(X_test)
preds_lgr = probs_lgr[:,1]
fpr_lgr, tpr_lgr, threshold_lgr = roc_curve(y_test, preds_lgr)
roc_auc_lgr = auc(fpr_lgr, tpr_lgr)

probs_knn = knn_clf.predict_proba(X_test)
preds_knn = probs_knn[:,1]
fpr_knn, tpr_knn, threshold_knn = roc_curve(y_test, preds_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr_lgr, tpr_lgr, 'g', label = 'AUC LGR = %0.2f' % roc_auc_lgr)
plt.plot(fpr_knn, tpr_knn, 'b', label = 'AUC KNN = %0.2f' % roc_auc_knn)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'m--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()






