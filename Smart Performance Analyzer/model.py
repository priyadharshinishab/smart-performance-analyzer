import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report, accuracy_score

# Load dataset
data = pd.read_csv('students.csv')
print("Dataset loaded successfully ✅")
print(data.head())

# ----- LINEAR REGRESSION PART -----
X = data[['hours_studied', 'attendance', 'internal_marks']]
y = data['final_marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

print("\n📈 LINEAR REGRESSION RESULTS")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# Visualize prediction
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Linear Regression: Actual vs Predicted Marks")
plt.show()

# ----- LOGISTIC REGRESSION PART -----
# Create pass/fail column
data['pass_fail'] = np.where(data['final_marks'] >= 50, 1, 0)

X_class = data[['hours_studied', 'attendance', 'internal_marks']]
y_class = data['pass_fail']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(Xc_train, yc_train)

yc_pred = log_reg.predict(Xc_test)

print("\n📊 LOGISTIC REGRESSION RESULTS")
print(f"Accuracy: {accuracy_score(yc_test, yc_pred):.2f}")
print(confusion_matrix(yc_test, yc_pred))
print(classification_report(yc_test, yc_pred))
