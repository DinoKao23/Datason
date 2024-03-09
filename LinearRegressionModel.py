import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

machine_df = pd.read_csv('machine_df.csv')

X = machine_df.drop('potential_total_value_of_award', axis = 1)
# Log transformation with python
y = np.log(machine_df[['potential_total_value_of_award']].values+1)
X_train,X_test, y_train, y_test = train_test_split(X, y,test_size= 0.2, random_state= 707)

k = 57
selector = SelectKBest(score_func=f_regression, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_indices = selector.get_support(indices=True)
selected_features = X_train.columns[selected_indices]


model = LinearRegression()
model.fit(X_train_selected, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test_selected)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

print(f"The actual values are: {np.exp(y_test[0])- 1}.The predict values are: {np.exp(y_pred[0])-1}")
print(f"The mean square error is: {mse}")
print(f"The rsquare error is: {r2_score(y_test, y_pred)}")