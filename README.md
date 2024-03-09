# 2024 LMU Datathon
This project aims to use predictive analytics to forecast the potential total value of awards granted by the Environmental Protection Agency (EPA). The focus is on aiding small, women-owned, and minority-owned businesses in securing EPA contract awards. Utilizing data from [USA Spending's Award Data Archive](https://www.usaspending.gov/download_center/award_data_archive), we've developed a pipeline consisting of data cleaning, preprocessing, and predictive modeling to achieve this objective.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation
This project requires the following packages:

- pandas version 1.5.3 or higher
- numpy version 1.24.4 or higher
- scikit-learn version 1.3.1 or higher

If you don't have these packages installed, you can install them using pip:

```
pip install pandas>=1.5.3 numpy>=1.24.4 scikit-learn>=1.3.1
```

## Usage
There be seperate into two part: Data Cleaning and Machine Learning
### Part 1: Data Cleaning and Preprocessing
`clean_data.py`

We cleaned and preprocessed EPA contract data, focusing on relevant columns, handling missing values and outliers, and preparing the dataset for machine learning models, resulting in a ready-to-use CSV file for predictive analysis.

The output is a cleaned and preprocessed dataset, saved as `machine_df.csv`, which is used in the subsequent predictive modeling phase.

### Part 2: Machine Learning
The second part of the project explores two modeling approaches to forecast the potential total value of EPA awards: regression and classification.

- Regression
We aim to predict the continuous value of the target variable using two distinct algorithms:
    - Linear Regression: This simple and fast algorithm serves as our baseline model, providing initial insights into the data's linear relationships. Detailed in `LinearRegressionModel.py`.
    - XGBoost: For enhanced precision and performance, we employ the XGBoost algorithm, known for its advanced capabilities in handling complex data structures. Detailed in `xgboost_regression.py`.

- Classification
    - Logistic Regression: We apply logistic regression to classify the potential value of EPA awards into specific categories. Detailed in `LogisticRegressionModel.py`.
    - Other models: Naive Bayes, SVC. Detailed in `Testing_Notebook.ipynb`.
Each model's performance is evaluated to determine the best approach for forecasting the value of EPA awards, considering the specific needs of small, women-owned, and minority-owned businesses.

## Contributing
Based on Alphabetical Order
- Tina Brauneck
- Tsai Lieh(Dino) Kao
- Riley Nickel
- Duong(Rachel) Pham

## License
Copyright (c) 2012-2024 Scott Chacon and others

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
