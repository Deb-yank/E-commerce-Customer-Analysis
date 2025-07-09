

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

df= pd.read_csv('ecommerce.csv')



df.describe()

df.info

df.head()

df

df.isnull().sum()

df.duplicated().sum()



sns.set(style='whitegrid')# Set the visual style
sns.set(style='whitegrid')

# Create the pairplot
plt.figure(figsize=(10, 8))
pair = sns.pairplot(
    df,
    kind='scatter',
    plot_kws={'alpha': 0.5, 's': 60, 'edgecolor': 'k'},
    corner=True,
    diag_kind='kde',
    palette='husl'
)

# Add a title
pair.fig.suptitle("Pairplot of Features", y=1.02, fontsize=16)

plt.show()

# Create a scatter plot for R&D Spend vs Profit
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Time on App', y='Yearly Amount Spent', color='teal', alpha=0.6, edgecolor='k')

X=df[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y= df['Yearly Amount Spent']

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

sns.set(style='whitegrid')
plt.figure(figsize=(8, 6))

sns.scatterplot(x=prediction, y=y_test, color='teal', alpha=0.6)

plt.xlabel("Predicted Profit")
plt.ylabel("Actual Profit")
plt.title("Predicted vs Actual Profit")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Reference line
plt.tight_layout()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Evaluation metrics
mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test,  prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, prediction)

# Print metrics
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

residuals= y_test-prediction
residuals

sns.displot(residuals, bins=10)

import pylab
import scipy.stats as stat
stat.probplot(residuals, dist='norm', plot= pylab)
pylab.show()

