import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# LOAD DATA
# -------------------------
data_path = r'C:\Users\Priyadharshini\OneDrive\Desktop\Smart Performance Analyzer\StudentsPerformance.csv'
data = pd.read_csv(data_path)

st.title("📊 Smart Performance Analyzer (Regression + Classification)")
st.write("This app predicts **Writing Score** (Linear Regression) and classifies **Pass/Fail** (Logistic Regression) using study-related features.")

st.subheader("🔹 Dataset Preview")
st.dataframe(data.head())

st.write("🧾 Columns found:")
st.write(list(data.columns))

# Check required columns
required_cols = ['math score', 'reading score', 'writing score']
missing_cols = [c for c in required_cols if c not in data.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

# -------------------------
# BASIC CLEANING / SETUP
# -------------------------
# Numeric conversion
for col in ['math score', 'reading score', 'writing score']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna()

# Add pass/fail column (writing score ≥ 40 → pass)
data['pass_fail'] = np.where(data['writing score'] >= 40, 1, 0)

st.success(f"✅ Cleaned dataset shape: {data.shape}")

# -------------------------
# LINEAR REGRESSION (Predict Writing Score)
# -------------------------
st.header("🎯 Linear Regression: Predict Writing Score")

X = data[['math score', 'reading score']]
y = data['writing score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

y_pred = lin_reg.predict(X_test)

st.write("**R² Score:**", round(r2_score(y_test, y_pred), 3))
st.write("**Mean Squared Error:**", round(mean_squared_error(y_test, y_pred), 3))

# Scatter plot
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.set_xlabel("Actual Writing Score")
ax.set_ylabel("Predicted Writing Score")
ax.set_title("Actual vs Predicted Writing Scores")
st.pyplot(fig)

# -------------------------
# LOGISTIC REGRESSION (Pass/Fail)
# -------------------------
st.header("⚙️ Logistic Regression: Predict Pass/Fail (Based on Writing Score ≥ 40)")

X_class = data[['math score', 'reading score']]
y_class = data['pass_fail']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_c, y_train_c)

y_pred_c = log_reg.predict(X_test_c)

st.write("**Accuracy:**", round(accuracy_score(y_test_c, y_pred_c), 3))

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c)
fig2, ax2 = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Confusion Matrix')
st.pyplot(fig2)

st.text("Classification Report:")
st.text(classification_report(y_test_c, y_pred_c))

# -------------------------
# USER INPUT PREDICTION
# -------------------------
st.header("🎓 Try Your Own Data")

math_score = st.number_input("Math Score", min_value=0, max_value=100, value=70)
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=75)

pred_writing = lin_reg.predict([[math_score, reading_score]])[0]
pred_pass = log_reg.predict([[math_score, reading_score]])[0]

st.subheader("📈 Predicted Writing Score: {:.2f}".format(pred_writing))
st.subheader("🎯 Prediction: {}".format("PASS ✅" if pred_pass == 1 else "FAIL ❌"))
