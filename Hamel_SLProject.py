import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import time


# Load the dataset
data_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

data = pd.read_csv(data_URL, names=column_names, na_values='?', skipinitialspace=True)

# Display the first few rows of the dataset
# print(data.head())
# print("\nLength of dataset: ", len(data))

# Check for missing values
# print('Columns with Missing Values:')
# print(data.isnull().sum(), '\n')

# Print missing data as a percentage of column length
missing_percentage = (data.isnull().sum() / len(data)) * 100
# print("Percentage of Missing Values per Column:")
# print(missing_percentage)

# Drop rows with missing values
data = data.dropna()

# print("Basic Statistics for Numeric Columns:")
# print(data.describe())

categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']
# for col in categorical_columns:
#     print("\nValue counts for", col, ":")
#     print(data[col].value_counts())

# total_entries = len(data)

# print("Top 5 most frequent entries for Work Class:")
# workclass_counts = data['workclass'].value_counts().head(5)
# workclass_percentages = (workclass_counts / total_entries) * 100
# print(pd.concat([workclass_counts, workclass_percentages], axis=1, keys=['Counts', 'Percentage']))
# print()

# print("Top 5 most frequent entries for Education:")
# education_counts = data['education'].value_counts().head(5)
# education_percentages = (education_counts / total_entries) * 100
# print(pd.concat([education_counts, education_percentages], axis=1, keys=['Counts', 'Percentage']))
# print()

# print("Top 5 most frequent entries for Occupation:")
# occupation_counts = data['occupation'].value_counts().head(5)
# occupation_percentages = (occupation_counts / total_entries) * 100
# print(pd.concat([occupation_counts, occupation_percentages], axis=1, keys=['Counts', 'Percentage']))
# print()

# print("Top 5 most frequent entries for Native Country:")
# native_country_counts = data['native-country'].value_counts().head(5)
# native_country_percentages = (native_country_counts / total_entries) * 100
# print(pd.concat([native_country_counts, native_country_percentages], axis=1, keys=['Counts', 'Percentage']))

# Define dictionaries to map numerical values back to categorical labels for each categorical column
workclass_mapping_reverse = {0: 'Private', 1: 'Self-emp-not-inc', 2: 'Self-emp-inc', 3: 'Federal-gov',
                             4: 'Local-gov', 5: 'State-gov', 6: 'Without-pay', 7: 'Never-worked'}
education_mapping_reverse = {0: 'Bachelors', 1: 'Some-college', 2: '11th', 3: 'HS-grad', 4: 'Prof-school',
                             5: 'Assoc-acdm', 6: 'Assoc-voc', 7: '9th', 8: '7th-8th', 9: '12th', 10: 'Masters',
                             11: '1st-4th', 12: '10th', 13: 'Doctorate', 14: '5th-6th', 15: 'Preschool'}
marital_status_mapping_reverse = {0: 'Married-civ-spouse', 1: 'Divorced', 2: 'Never-married', 3: 'Separated',
                                  4: 'Widowed', 5: 'Married-spouse-absent', 6: 'Married-AF-spouse'}
occupation_mapping_reverse = {0: 'Tech-support', 1: 'Craft-repair', 2: 'Other-service', 3: 'Sales', 4: 'Exec-managerial',
                              5: 'Prof-specialty', 6: 'Handlers-cleaners', 7: 'Machine-op-inspct', 8: 'Adm-clerical',
                              9: 'Farming-fishing', 10: 'Transport-moving', 11: 'Priv-house-serv', 12: 'Protective-serv',
                              13: 'Armed-Forces'}
relationship_mapping_reverse = {0: 'Wife', 1: 'Own-child', 2: 'Husband', 3: 'Not-in-family', 4: 'Other-relative',
                                5: 'Unmarried'}
race_mapping_reverse = {0: 'White', 1: 'Asian-Pac-Islander', 2: 'Amer-Indian-Eskimo', 3: 'Other', 4: 'Black'}
sex_mapping_reverse = {0: 'Female', 1: 'Male'}
native_country_mapping_reverse = {0: 'United-States', 1: 'Cambodia', 2: 'England', 3: 'Puerto-Rico', 4: 'Canada',
                                  5: 'Germany', 6: 'Outlying-US(Guam-USVI-etc)', 7: 'India', 8: 'Japan', 9: 'Greece',
                                  10: 'South', 11: 'China', 12: 'Cuba', 13: 'Iran', 14: 'Honduras', 15: 'Philippines',
                                  16: 'Italy', 17: 'Poland', 18: 'Jamaica', 19: 'Vietnam', 20: 'Mexico', 21: 'Portugal',
                                  22: 'Ireland', 23: 'France', 24: 'Dominican-Republic', 25: 'Laos', 26: 'Ecuador',
                                  27: 'Taiwan', 28: 'Haiti', 29: 'Columbia', 30: 'Hungary', 31: 'Guatemala', 32: 'Nicaragua',
                                  33: 'Scotland', 34: 'Thailand', 35: 'Yugoslavia', 36: 'El-Salvador', 37: 'Trinadad&Tobago',
                                  38: 'Peru', 39: 'Hong', 40: 'Holand-Netherlands'}
income_mapping_reverse = {0: '<=50K', 1: '>50K'}

# Print the dictionaries
# print("Workclass Mapping:")
# print(workclass_mapping_reverse)
# print("\nEducation Mapping:")
# print(education_mapping_reverse)
# print("\nMarital Status Mapping:")
# print(marital_status_mapping_reverse)
# print("\nOccupation Mapping:")
# print(occupation_mapping_reverse)
# print("\nRelationship Mapping:")
# print(relationship_mapping_reverse)
# print("\nRace Mapping:")
# print(race_mapping_reverse)
# print("\nSex Mapping:")
# print(sex_mapping_reverse)
# print("\nNative Country Mapping:")
# print(native_country_mapping_reverse)
# print("\nIncome Mapping:")
# print(income_mapping_reverse)


# # Convert categorical variables to numerical using one-hot encoding
data_encoded = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

print(data_encoded.head())
print(len(data_encoded))

# Split data into features and target variables
x = data_encoded.drop('income_>50K', axis=1)  # Features
y = data_encoded['income_>50K']  # Target variable

# Split into test / training sets at 20/80 split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# #### DECISION TREE ####

# # Start time
# start_time = time.time()

# dt_classifier = DecisionTreeClassifier(random_state=42)

# dt_classifier.fit(x_train, y_train)

# dt_pred = dt_classifier.predict(x_test)

# dt_accuracy = accuracy_score(y_test, dt_pred)
# print()
# print("Initial Decision Tree Accuracy:", dt_accuracy)

# dt_param_grid = {'max_depth': [3, 5, 7, 9, None], 'min_samples_split': [2, 5, 10]}
# dt_grid_search = GridSearchCV(dt_classifier, dt_param_grid, cv=5)

# dt_grid_search.fit(x_train, y_train)

# dt_best_params = dt_grid_search.best_params_
# print('Best Parameters for Decision Tree')
# print(dt_best_params)

# dt_classifier = DecisionTreeClassifier(**dt_best_params, random_state=42)

# dt_classifier.fit(x_train, y_train)

# dt_pred = dt_classifier.predict(x_test)

# dt_accuracy = accuracy_score(y_test, dt_pred)

# print()
# print("Post Tune Decision Tree Accuracy:", dt_accuracy)

# # End time
# end_time = time.time()

# # Elapsed time
# elapsed_time = end_time - start_time
# print()
# print("Decision Tree Elapsed time:", elapsed_time, "seconds")
# print()

# # Evaluate the best model
# print("Decision Tree Classification Report:")
# print(classification_report(y_test, dt_pred))
# print("Decision Tree Confusion Matrix:")
# print(confusion_matrix(y_test, dt_pred))
# print("Decision Tree Accuracy Score:", accuracy_score(y_test, dt_pred))


# #### K NEAREST NEIGHBORS ####

# # Start time
# start_time = time.time()

# knn_classifier = KNeighborsClassifier()
# knn_classifier.fit(x_train, y_train)
# knn_pred = knn_classifier.predict(x_test)
# knn_accuracy = accuracy_score(y_test, knn_pred)
# print()
# print("KNN Initial Accuracy:", knn_accuracy)
# print()

# knn_param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
# knn_grid_search = GridSearchCV(knn_classifier, knn_param_grid, cv=5)
# knn_grid_search.fit(x_train, y_train)

# knn_grid_search.fit(x_train, y_train)

# knn_best_params = knn_grid_search.best_params_

# knn_classifier = KNeighborsClassifier(**knn_best_params)

# knn_classifier.fit(x_train, y_train)

# knn_pred = knn_classifier.predict(x_test)
# knn_accuracy = accuracy_score(y_test, knn_pred)

# print("KNN Accuracy Post Tune:", knn_accuracy)
# print()
# print("Best parameters for KNN:", knn_grid_search.best_params_)

# # End time
# end_time = time.time()

# # Elapsed time
# elapsed_time = end_time - start_time
# print()
# print("KNN Elapsed time:", elapsed_time, "seconds")
# print()

# # Evaluate the best model
# print("KNN Classification Report:")
# print(classification_report(y_test, knn_pred))
# print("KNN Confusion Matrix:")
# print(confusion_matrix(y_test, knn_pred))
# print("KNN Accuracy Score:", accuracy_score(y_test, knn_pred))

# #### RANDOM FOREST ####

# # Set up hyperparameter tuning using GridSearchCV
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 4, 6],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False]
# }

# # Start time
# start_time = time.time()
# # Create a Random Forest classifier
# rf = RandomForestClassifier(random_state=42)

# rf.fit(x_train, y_train)

# rf_predict = rf.predict(x_test)

# rf_accuracy = accuracy_score(y_test, rf_predict)

# print()
# print("Initial Random Forest Accuracy:", rf_accuracy)

# # Initialize GridSearchCV
# grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)

# # Run the grid search
# grid_search.fit(x_train, y_train)


# # Best parameters
# print()
# print("Best Hyperparameters:")
# print(grid_search.best_params_)

# # Best model
# best_rf = grid_search.best_estimator_

# # Make predictions with the best model
# y_pred = best_rf.predict(x_test)

# # End time
# end_time = time.time()

# # Elapsed time
# elapsed_time = end_time - start_time
# print()
# print("Random Forest Elapsed time:", elapsed_time, "seconds")
# print()

# # Evaluate the best model
# print("Random Forest Classification Report:")
# print(classification_report(y_test, y_pred))
# print("Random Forest Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("Post Tune Random Forest Accuracy Score:", accuracy_score(y_test, y_pred))


# ### ROC Curve ###

# # Get the probabilities for each model
# dt_probs = dt_classifier.predict_proba(x_test)[:, 1]
# rf_probs = best_rf.predict_proba(x_test)[:, 1]
# knn_probs = knn_classifier.predict_proba(x_test)[:, 1]

# # Generate ROC curves
# fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
# fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
# fpr_knn, tpr_knn, _ = roc_curve(y_test, knn_probs)

# # Compute AUC for each model
# auc_dt = roc_auc_score(y_test, dt_probs)
# auc_rf = roc_auc_score(y_test, rf_probs)
# auc_knn = roc_auc_score(y_test, knn_probs)

# # Plotting the ROC curves
# plt.figure(figsize=(10, 6))
# plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {auc_dt:.2f})")
# plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")
# plt.plot(fpr_knn, tpr_knn, label=f"k-NN (AUC = {auc_knn:.2f})")
# plt.plot([0, 1], [0, 1], linestyle='--', color='grey', linewidth=2, label='Random Predictor')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curves for Decision Tree, Random Forest, and k-NN")
# plt.legend(loc="best")
# plt.show()

# ### FEATURE IMPORTANCE ###

# ### Decision Tree ###

# dt_importance = dt_classifier.feature_importances_
# feature_names = x_train.columns.tolist()

# # Sort features by importance
# sorted_indices = np.argsort(dt_importance)[::-1]
# top_10_indices = sorted_indices[:10]
# top_10_features = [feature_names[i] for i in top_10_indices]
# top_10_importance = dt_importance[top_10_indices]

# # Plot the top 10 features
# plt.figure(figsize=(10, 6))
# plt.bar(top_10_features, top_10_importance)
# plt.xlabel("Features")
# plt.ylabel("Importance")
# plt.title("Top 10 Feature Importance - Decision Tree")
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()


# ### Random Forests ###

# rf_importance = best_rf.feature_importances_

# # Sort features by importance
# sorted_indices = np.argsort(rf_importance)[::-1]
# top_10_indices = sorted_indices[:10]
# top_10_features = [feature_names[i] for i in top_10_indices]
# top_10_importance = rf_importance[top_10_indices]

# # Plot the top 10 features
# plt.figure(figsize=(10, 6))
# plt.bar(top_10_features, top_10_importance)
# plt.xlabel("Features")
# plt.ylabel("Importance")
# plt.title("Top 10 Feature Importance - Random Forest")
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()