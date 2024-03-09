import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import json

machine_df = pd.read_csv('machine_df.csv')
X = machine_df.drop('potential_total_value_of_award', axis = 1)
y = machine_df[['potential_total_value_of_award']]

# split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state= 707)
y_train = y_train.squeeze()
y_test = y_test.squeeze()
print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# create a function to do feature selection
def select_features(X_train, y_train, X_test, k):
    selector = SelectKBest(f_regression, k=k)
    x_train_selected = selector.fit_transform(X_train, y_train)
    x_test_selected = selector.transform(X_test)
    
    mask = selector.get_support()
    selected_features = X_train.columns[mask]
    selected_features_list = selected_features.tolist()
    
    return x_train_selected, x_test_selected, selected_features_list

################## FIND THE BEST XGBOOST MODEL ##################

n_features = range(1, len(X.columns)+1)
test_rmse_xgb = []
test_mse_xgb = []
test_r2_xgb = []
dict_selected_features_xgb = {}
dict_best_params_xgb = {}
for k in tqdm(n_features):
    x_train_selected, x_test_selected, selected_features = select_features(X_train, y_train, X_test, k)
    dict_selected_features_xgb[k] = selected_features
    
    xgb = XGBRegressor(random_state=42)
    
    # Define hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_child_weight': [1, 3]
    }

    # Setup GridSearchCV
    grid_search_xgb = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

    # Fit the model
    grid_search_xgb.fit(x_train_selected, y_train.values.ravel())

    # Predict on the test set with the best parameters
    y_pred_xgb = grid_search_xgb.predict(x_test_selected)

    # Performance metrics
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    rmse_xgb = mse_xgb ** 0.5
    r2_xgb = r2_score(y_test, y_pred_xgb)
    
    dict_best_params_xgb[k] = grid_search_xgb.best_params_
    dict_best_params_xgb[k]['r2'] = r2_xgb
    dict_best_params_xgb[k]['rmse'] = rmse_xgb
    dict_best_params_xgb[k]['mse'] = mse_xgb

    test_rmse_xgb.append(rmse_xgb)
    test_mse_xgb.append(mse_xgb)
    test_r2_xgb.append(r2_xgb)
    
plt.plot(n_features, test_r2_xgb, label='R2')
plt.savefig('xgboost_regression.png')
plt.close()

# save 2 dicts to json with nice formatting
with open('dict_selected_features_xgb.json', 'w') as f:
    f.write(json.dumps(dict_selected_features_xgb, indent=4))
    
with open('dict_best_params_xgb.json', 'w') as f:
    f.write(json.dumps(dict_best_params_xgb, indent=4))