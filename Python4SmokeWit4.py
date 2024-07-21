#Linear regression python code

from readline import redisplay
import pandas as pd
import statsmodels.api as sm

# Create the DataFrame

data = {
    'Gender': ['male', 'female', 'female', 'male', 'female', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'male', 'male', 'male', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'female', 'female', 'male', 'male', 'female', 'male', 'male', 'female', 'male', 'male', 'male', 'male', 'female', 'male', 'male', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'male', 'male', 'female', 'female', 'female', 'female', 'female', 'female', 'male', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'female', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'female', 'male', 'male', 'female', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'male', 'male', 'female', 'male', 'male', 'female', 'male', 'male', 'female', 'male', 'male', 'female', 'male', 'male', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'male', 'male', 'female', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'male', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female', 'female', 'male', 'female', 'female', 'female', 'female', 'female', 'male', 'female', 'female', 'male', 'female'],
     'Age': [57, 37, 56, 46, 52, 64, 35, 24, 42, 43, 37, 60, 59, 51, 37, 56, 62, 47, 39, 56, 54, 58, 27, 29, 28, 57, 49, 63, 53, 43, 56, 24, 59, 34, 25, 62, 25, 62, 46, 26, 58, 43, 54, 45, 45, 42, 24, 54, 58, 24, 30, 24, 38, 62, 60, 59, 44, 42, 26, 45, 54, 44, 51, 48, 35, 63, 36, 47, 31, 54, 48, 41, 44, 45, 40, 51, 58, 28, 33, 42, 61, 61, 57, 34, 23, 30, 64, 28, 38, 43, 55, 56, 60, 47, 53, 55, 25, 45, 23, 61, 47, 23, 50, 63, 31, 63, 54, 56, 63, 44, 40, 54, 29, 48, 37, 60, 55, 61, 42, 41, 31, 25, 61, 31, 37, 52, 47, 32, 59, 23, 34, 39, 38, 31, 54, 38, 44, 32, 60, 53, 54, 30, 33, 33, 36, 29, 31, 45, 64, 38, 32, 33, 64, 58, 30, 28, 44, 46, 62, 38, 23, 53, 58, 49, 56, 54, 63, 61, 39, 38, 57, 36, 41, 29, 54, 58, 52, 24, 58, 64, 58, 26, 32, 27, 33, 30, 25, 36, 43, 50, 63, 54, 28, 42, 40, 36, 27, 23, 49, 46],
  'Smoke_Free': [18, 18, 12, 5, 13, 7, 19, 8, 14, 7, 6, 5, 21, 10, 17, 11, 2, 11, 12, 16, 6, 3, 9, 15, 2, 1, 5, 12, 19, 18, 5, 16, 3, 18, 11, 8, 10, 13, 9, 19, 1, 3, 10, 20, 16, 19, 1, 8, 4, 14, 21, 20, 18, 9, 7, 3, 18, 15, 15, 17, 12, 14, 5, 14, 19, 8, 19, 12, 12, 9, 20, 2, 14, 21, 7, 7, 17, 10, 10, 9, 13, 11, 20, 15, 10, 4, 21, 1, 7, 14, 16, 7, 12, 21, 4, 19, 14, 11, 3, 1, 20, 13, 12, 3, 12, 19, 21, 3, 19, 4, 21, 6, 4, 8, 7, 9, 18, 19, 21, 7, 2, 1, 4, 20, 4, 14, 1, 3, 6, 19, 20, 2, 17, 17, 8, 12, 2, 18, 5, 6, 8, 12, 8, 13, 6, 1, 15, 10, 8, 10, 13, 15, 20, 8, 21, 5, 9, 20, 21, 4, 17, 9, 3, 6, 19, 15, 13, 14, 16, 14, 17, 18, 1, 12, 20, 17, 13, 20, 14, 12, 10, 19, 1, 9, 1, 10, 9, 6, 5, 11, 15, 14, 13, 10, 1, 15, 18, 1, 11, 9],
  'Wt_Before': [80.7, 61.9, 69.4, 89.2, 67.1, 82.7, 75.5, 82.9, 77.6, 59.2, 61.4, 68.3, 81.4, 79.5, 77.3, 84.2, 68.1, 67.5, 82.9, 76.1, 78.3, 67.7, 78.2, 78.5, 62.6, 65.4, 65.5, 74, 80.3, 58.9, 77.2, 69.6, 69.3, 64.6, 76.8, 81.9, 66.1, 76.3, 81.5, 65.2, 74.3, 75.2, 76.7, 71.4, 58.4, 79.3, 79.7, 64.3, 62.5, 76.2, 74.5, 78.4, 65.8, 67.6, 76, 64.9, 85.2, 79.8, 61.8, 68.3, 72.7, 70, 77.4, 70.9, 75.9, 64.5, 66.8, 82.1, 80, 65.8, 59.6, 64.8, 78.6, 71.1, 66.7, 74.2, 64.3, 76.6, 61, 78.4, 64.9, 76.6, 63.6, 65.6, 82.9, 68, 66.3, 70.7, 78.6, 84.8, 62.9, 62.3, 68.6, 62.5, 64.9, 65.4, 76.1, 74.2, 64.8, 67.8, 75.6, 72, 75.6, 66.1, 72.7, 64.2, 113.8, 79.1, 67.5, 67.6, 64.7, 65.5, 81, 64.2, 74.2, 61, 109.3, 59.3, 70.9, 58.2, 63.7, 71.9, 78, 79, 66.5, 68.5, 62.8, 83.5, 56, 73.3, 65.1, 80.7, 78.8, 80.4, 78.1, 56.8, 82.6, 85.4, 63, 85.9, 73.7, 65.4, 82.2, 58, 65.5, 78, 75.6, 65.3, 80.7, 60.3, 66, 79.5, 63.5, 81.4, 71.7, 63.3, 77.8, 83, 66.4, 75, 76.3, 63.1, 66.8, 63.1, 79.9, 64.4, 110.7, 74.7, 66.9, 63.7, 79.5, 64.8, 62.8, 77, 67, 77.8, 67.5, 70.9, 79.7, 77.9, 63, 78.5, 59.6, 82.7, 60.5, 75.4, 80.9, 61.2, 62.1, 102.6, 65.8, 66.8, 62.6, 67.8, 64.8, 82.4, 66.5, 66.1, 79.1, 66.4],
  'Wt_After': [79.4, 62.5, 68.4, 85.1, 65.2, 81.6, 73.6, 80.2, 74.9, 59.8, 63, 66.3, 78.2, 77.4, 76, 81.1, 68.5, 66.2, 78.5, 74.3, 76.6, 68.2, 75.6, 76.8, None, 65.3, 64.7, 73.4, 77.6, 60.2, 76.5, 69.3, 70.8, 66.5, 75.6, 79.5, 65.6, 73.5, 78.3, 63.8, 74.1, 73.7, 73.7, 70.5, 59.4, 79.1, 77.1, 63.4, 62.7, 76.3, None, 76.8, 66.3, 67.9, 73.2, 66.7, 81.6, 77.9, 62.6, 70.4, 72.2, 69.2, 74, 68.8, 74.4, 63.4, 65.1, 79, 78.9, 65.8, 61.9, 66, 75.4, 68.6, 66.6, 71.6, 63.1, 74.9, 61.9, 76.7, 65.1, 74.9, 65.3, 65, 79.9, 68.2, 67.3, 69.7, 77.6, 81, 62, 63.8, 69.5, 63.1, 67.1, 66.5, 75, 72.8, 64.1, 66.7, 74.8, 71.4, 74.4, 64.1, 71, 62.7, None, 75.2, 65.6, 67, 64.7, 65.6, 78, 64, None, 64.2, 102.6, 60.6, 70.2, 59.8, 62.2, 70.4, 74.7, 74.1, 67.9, 69, None, 80, 57, 71.2, 66, 77.7, 76.7, 75.8, 76, 57.8, 79.3, 82.2, 63.9, 81.5, 72.2, 65.2, 79.8, 59.2, 65, 75.6, 73.4, 66.1, 78, 61.1, 65.6, 77.4, 64, 79.5, 72.1, 64.9, 75.4, 82.1, 67, 72.4, 74.6, 64.8, 67.8, 64.3, 77, 67.1, None, 72, 67.2, 65.6, 76.2, 64.1, 63, 76.1, 65.8, 76, 66.5, 72.7, 78.3, 76.7, 62.2, 77.2, 60.7, 78.4, 61.6, 72.4, 79.1, 62.8, 64.8, 95.6, 67, 65.9, 65.2, 68.4, 65.7, 79.3, 67.2, 66.8, 77.6, 66.8]
}

df = pd.DataFrame(data)

# Convert categorical variable 'Gender' to numeric
df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})

# Identify missing values
missing = df['Wt_After'].isnull()

# Fit regression model
X = df[['Gender', 'Age', 'Smoke_Free', 'Wt_Before']]
X = sm.add_constant(X)  # Adds a constant term to the predictor
y = df['Wt_After']
model = sm.OLS(y[~missing], X[~missing]).fit()

# Predict missing values
df.loc[missing, 'Wt_After'] = model.predict(X[missing])

print(df)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
# All dataframes hereafter reflect these changes.
print(df)
