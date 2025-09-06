# ----------------------------
# Step 0: Import Libraries
# ----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Import models for each group member
from sklearn.neighbors import KNeighborsClassifier # Model 1: K-Nearest Neighbors
from sklearn.svm import SVC # Model 2: Support Vector Machine
from sklearn.ensemble import RandomForestClassifier # Model 3: Random Forest

# Import metrics for evaluation
from sklearn.metrics import classification_report, accuracy_score

import joblib

print("Libraries imported successfully!")

# ----------------------------
# Step 1: Load and Understand the Dataset (Assignment Part 1c)
# ----------------------------
# You can download the dataset from Kaggle: https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data
# Make sure you have the 'Train.csv' file in the same directory as this script.
try:
    df = pd.read_csv('Train.csv')
    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
except FileNotFoundError:
    print("Error: 'Train.csv' not found. Please download it from Kaggle and place it in the correct directory.")
    exit() # Exit the script if the data isn't there.

# ----------------------------
# Step 2: Data Pre-processing and Feature Engineering (Assignment Part 1d)
# ----------------------------
print("\n--- Starting Data Pre-processing ---")

# a) Create the Target Variable: Classify sales into categories
# The original dataset has a continuous 'Item_Outlet_Sales' column.
# We need to convert it into 'Low', 'Medium', 'High' classes for our classification problem.
sales_bins = [0, 1500, 4000, np.inf]
sales_labels = ['Low', 'Medium', 'High']
df['Sales_Class'] = pd.cut(df['Item_Outlet_Sales'], bins=sales_bins, labels=sales_labels, right=False)

# b) Handle Missing Values
# Impute missing 'Item_Weight' with the mean weight.
df['Item_Weight'].fillna(df['Item_Weight'].mean(), inplace=True)

# Impute missing 'Outlet_Size' with the mode (most frequent value).
df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0], inplace=True)
print("Missing values handled.")

# c) Correct Inconsistent Categorical Data
# The 'Item_Fat_Content' has inconsistent values like 'low fat' and 'LF'. Let's standardize them.
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
    'low fat': 'Low Fat',
    'LF': 'Low Fat',
    'reg': 'Regular'
})
# Your app also maps 'Non-Edible' to 'Low Fat' during prediction, so we can do that here for consistency.
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'Non-Edible': 'Low Fat'})
print("Item_Fat_Content standardized.")

# d) Define Features (X) and Target (y)
# We will drop the original sales column and identifiers that are not useful for prediction.
X = df.drop(['Item_Outlet_Sales', 'Sales_Class', 'Item_Identifier', 'Outlet_Identifier'], axis=1)
y = df['Sales_Class']

# Get the list of columns for the Streamlit app later
model_columns = list(X.columns)

# e) Split Data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")


# ----------------------------
# Step 3: Building a Pre-processing Pipeline
# ----------------------------
# This pipeline will handle categorical and numerical features separately.
# It makes the process clean and prevents data leakage from the test set.

# Identify categorical and numerical columns
numerical_features = X.select_dtypes(include=np.number).columns
categorical_features = X.select_dtypes(include='object').columns

# Create a pre-processing pipeline for numerical features (impute then scale)
numerical_transformer = StandardScaler()

# Create a pre-processing pipeline for categorical features (one-hot encode)
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any)
)


# ----------------------------
# Step 4: Train and Evaluate Models (Assignment Part 1e & 1f)
# ----------------------------
print("\n--- Training and Evaluating Models ---")

# Define the models to be trained
models = {
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine (SVM)": SVC(kernel='rbf', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_accuracy = 0

for name, model in models.items():
    # Create the full pipeline: pre-process data then train the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\n--- Results for {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)

    # Store results and check if this is the best model so far
    results[name] = accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model_pipeline = pipeline

print(f"\nBest performing model is {best_model_name} with an accuracy of {best_accuracy:.4f}")


# ----------------------------
# Step 5: Save the Best Model and Assets for Your Streamlit App
# ----------------------------
# Now, let's save the assets required by your `app.py`.
# Your app needs three things: the model, the scaler, and the column names.
# Our 'pipeline' object contains the trained model and the fitted scaler/encoder.

# We will need to re-create the final pre-processing steps outside the pipeline
# to save the scaler and column names in the exact format your app needs.

# 1. One-hot encode the full feature set to get the final column names
X_encoded = pd.get_dummies(X)
model_final_columns = list(X_encoded.columns) # These are the columns your model was trained on

# 2. Fit the scaler on the full encoded data
final_scaler = StandardScaler()
final_scaler.fit(X_encoded)

# 3. Retrain the best model on the FULL pre-processed dataset
# This is a common practice to give the final model as much data as possible.
best_model_final = models[best_model_name] # Get a fresh instance of the best model
X_encoded_scaled = final_scaler.transform(X_encoded)
best_model_final.fit(X_encoded_scaled, y)
print(f"\nRetrained the best model ({best_model_name}) on the full dataset.")


# 4. Save the assets into a single joblib file
assets_to_save = {
    'model': best_model_final,
    'scaler': final_scaler,
    'columns': model_final_columns
}

joblib.dump(assets_to_save, 'sales_classifier.joblib')

print("\n✅ Successfully saved 'sales_classifier.joblib'. You can now run your Streamlit app!")