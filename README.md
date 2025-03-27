# DENTIST-PROJECT
Using Dental Metrics to Predict Gender
import pandas as pd

# Load the dataset
file_path = "/mnt/data/Dentistry Dataset.csv"

# Read the CSV file
df = pd.read_csv(file_path)

# Display basic information about the dataset
df_info = df.info()
df_head = df.head()

# Check for missing values
missing_values = df.isnull().sum()

df_info, df_head, missing_values

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Drop the "Sample ID" column as it is completely empty
df = df.drop(columns=["Sample ID"])

# Encode the 'Gender' column using Label Encoding (Female = 0, Male = 1)
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])

# Normalize all numerical columns except "Sl No" (index column)
scaler = MinMaxScaler()
numerical_cols = df.columns.difference(["Sl No"])  # Exclude "Sl No" from scaling
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display the first few rows after preprocessing
df.head()

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Drop the "Sample ID" column as it is completely empty
df = df.drop(columns=["Sample ID"])

# Encode the 'Gender' column using Label Encoding (Female = 0, Male = 1)
label_encoder = LabelEncoder()
df["Gender"] = label_encoder.fit_transform(df["Gender"])

# Normalize all numerical columns except "Sl No" (index column)
scaler = MinMaxScaler()
numerical_cols = df.columns.difference(["Sl No"])  # Exclude "Sl No" from scaling
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Display the first few rows after preprocessing
df.head()

import matplotlib.pyplot as plt
import seaborn as sns

# Set figure size
plt.figure(figsize=(12, 8))

# Create a heatmap to visualize correlations between variables
sns.heatmap(df.drop(columns=["Sl No"]).corr(), annot=True, cmap="coolwarm", fmt=".2f")

# Show the plot
plt.title("Feature Correlation Heatmap")
plt.show()

from sklearn.model_selection import train_test_split

# Drop highly correlated features to avoid multicollinearity
features_to_drop = ["right canine width casts", "left canine width casts", "right canine index casts", "left canine index casts"]
df = df.drop(columns=features_to_drop)

# Define independent (X) and dependent (Y) variables
X = df.drop(columns=["Sl No", "Gender"])  # Features (excluding Sl No and Gender)
Y = df["Gender"]  # Target variable

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Display the shapes of training and testing sets
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

# Initialize models
log_reg = LogisticRegression()
dtree = DecisionTreeClassifier()
rf = RandomForestClassifier()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train models
log_reg.fit(X_train, Y_train)
dtree.fit(X_train, Y_train)
rf.fit(X_train, Y_train)
xgb.fit(X_train, Y_train)

# Predictions
log_reg_pred = log_reg.predict(X_test)
dtree_pred = dtree.predict(X_test)
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)

# Model Evaluation (Accuracy, AUC-ROC Score)
model_results = {
    "Logistic Regression": {
        "Accuracy": accuracy_score(Y_test, log_reg_pred),
        "ROC-AUC": roc_auc_score(Y_test, log_reg_pred)
    },
    "Decision Tree": {
        "Accuracy": accuracy_score(Y_test, dtree_pred),
        "ROC-AUC": roc_auc_score(Y_test, dtree_pred)
    },
    "Random Forest": {
        "Accuracy": accuracy_score(Y_test, rf_pred),
        "ROC-AUC": roc_auc_score(Y_test, rf_pred)
    },
    "XGBoost": {
        "Accuracy": accuracy_score(Y_test, xgb_pred),
        "ROC-AUC": roc_auc_score(Y_test, xgb_pred)
    }
}

model_results

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve

# Create subplots for confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Function to plot confusion matrix
def plot_conf_matrix(y_true, y_pred, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Female", "Male"], yticklabels=["Female", "Male"], ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

# Plot confusion matrices for all models
plot_conf_matrix(Y_test, log_reg_pred, "Logistic Regression", axes[0, 0])
plot_conf_matrix(Y_test, dtree_pred, "Decision Tree", axes[0, 1])
plot_conf_matrix(Y_test, rf_pred, "Random Forest", axes[1, 0])
plot_conf_matrix(Y_test, xgb_pred, "XGBoost", axes[1, 1])

# Show the plots
plt.tight_layout()
plt.show()
