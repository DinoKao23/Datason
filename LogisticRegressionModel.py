import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split

machine_df = pd.read_csv('machine_df.csv')

range = [-1, 120000, 310000, np.inf]
group_name = ['0-120k', '120k-310k', '310k+']

machine_df['income_level'] = pd.cut(machine_df['potential_total_value_of_award'], bins = range, labels = group_name)

machine_df['income_level'].value_counts()

X = machine_df.drop(['potential_total_value_of_award','income_level'], axis = 1)
y = machine_df[['income_level']]

X.reset_index().drop('index', axis = 1, inplace=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=707, stratify= y)

# Initialize the logistic regression model
logistic_model = LogisticRegression(multi_class = 'multinomial', C=0.6)

# Train the logistic regression model on the training data
logistic_model.fit(X_train, y_train)

# Predict the target variable on the testing set
y_pred = logistic_model.predict(X_test)

# Evaluate the performance of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy,3))

# Additional evaluation metrics
# print(classification_report(y_test, y_pred))