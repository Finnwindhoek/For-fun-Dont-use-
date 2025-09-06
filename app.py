import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Sales Predictor Pro",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Load Assets
# ----------------------------
# Use st.cache_resource to load these heavy assets only once.
@st.cache_resource
def load_assets():
    """Loads all necessary assets from the joblib file."""
    try:
        # Load the dictionary of assets from the single .joblib file
        assets = joblib.load("sales_classifier.joblib")
        return assets
    except FileNotFoundError:
        st.error("Asset file 'sales_classifier.joblib' not found. Please ensure it's in the same directory as the app.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the assets: {e}")
        return None

# Load all assets from the file
assets = load_assets()

# The app cannot run if the assets are not loaded.
if assets is None:
    st.warning("Application assets could not be loaded. Please check the logs.")
    st.stop() # Stop the script from running further.

# Extract each asset into its own variable for easier access
models = assets.get('models')
scaler = assets.get('scaler')
model_columns = assets.get('columns')
imputation_values = assets.get('imputation_values')

# ----------------------------
# Prediction Function
# ----------------------------
def predict(input_df, model, scaler, columns, imputation_vals):
    """
    Preprocesses the input data and returns model predictions.
    This function now correctly uses the saved imputation values.
    """
    processed_df = input_df.copy()

    # Standardize 'Item_Fat_Content' to match the training data's categories
    processed_df['Item_Fat_Content'] = processed_df['Item_Fat_Content'].replace({
        'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular', 'Non-Edible': 'Low Fat'
    })

    # Fill any missing values using the exact values saved from the training process
    if 'Item_Weight_median' in imputation_vals:
      processed_df['Item_Weight'].fillna(imputation_vals['Item_Weight_median'], inplace=True)

    cat_cols = processed_df.select_dtypes(include=['object']).columns.tolist()
    for col in cat_cols:
        if f"{col}_mode" in imputation_vals:
            processed_df[col].fillna(imputation_vals[f"{col}_mode"], inplace=True)

    # One-hot encode and align columns to match the model's training data
    processed_encoded = pd.get_dummies(processed_df, drop_first=True)
    processed_aligned = processed_encoded.reindex(columns=columns, fill_value=0)

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(processed_aligned)

    # Make predictions
    predictions = model.predict(scaled_features)
    return predictions

# ----------------------------
# Sidebar UI
# ----------------------------
st.sidebar.image("https://www.oneshop.co.zw/wp-content/uploads/2021/11/undraw_add_to_cart_re_wrdo.svg", use_column_width=True)
st.sidebar.title("👨‍💻 Model Controls")
st.sidebar.markdown("Select your preferred model and navigation page.")

# FEATURE: Model Selection Dropdown
model_names = list(models.keys())
selected_model_name = st.sidebar.selectbox(
    "Select Model (ANN, SVM, KNN)",
    model_names,
    # Default to SVM_linear if it exists, otherwise the first model
    index=model_names.index("SVM_linear") if "SVM_linear" in model_names else 0
)
# Get the actual model object from the dictionary
selected_model = models.get(selected_model_name)

# Highlight the best-performing model as identified in your analysis
if selected_model_name == "SVM_linear":
    st.sidebar.success("💡 **Best performing model!**")

# Use a radio button for cleaner navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "📊 Single Prediction", "📂 Batch Prediction"])

# ----------------------------
# Main Page UI
# ----------------------------

if page == "🏠 Home":
    st.title("🛒 Welcome to Sales Predictor Pro!")
    st.markdown("""
        This application is an interactive tool to predict sales performance for grocery items.
        It was built to demonstrate the practical application of various machine learning classification models.

        ### ✨ Key Features:
        - **Multiple Model Selection:** Choose from different trained models like **SVM, KNN, and ANN** right from the sidebar.
        - **Real-time Predictions:** Get instant sales class predictions ('Low', 'Medium', or 'High').
        - **Single & Batch Mode:** Predict a single item's performance or upload a CSV for bulk predictions.

        ### 🚀 How to Get Started:
        1.  **Select a Model:** Use the dropdown in the sidebar to choose the algorithm you want to use.
        2.  **Navigate:** Select either 'Single Prediction' or 'Batch Prediction'.
        3.  **Input Data:** Provide the item details and get your result!
    """)
    st.info(f"You have currently selected the **{selected_model_name}** model.", icon="🤖")

elif page == "📊 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    st.markdown(f"You are currently using the **`{selected_model_name}`** model.")

    with st.form("single_prediction_form"):
        st.subheader("Enter Product & Outlet Details")
        col1, col2 = st.columns(2)
        with col1:
            item_weight = st.number_input("Item Weight (kg)", min_value=0.1, value=12.5, format="%.2f")
            item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
            item_visibility = st.slider("Item Visibility (%)", 0.0, 100.0, 6.0, 0.1) / 100.0 # Converted to ratio
            item_type = st.selectbox("Item Type", ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood'])

        with col2:
            item_mrp = st.number_input("Item MRP (Price)", min_value=10.0, value=140.0, format="%.2f")
            outlet_establishment_year = st.selectbox("Outlet Establishment Year", sorted(list(range(1985, 2010))), index=17)
            outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
            outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
            outlet_type = st.selectbox("Outlet Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])

        submitted = st.form_submit_button("🚀 Predict Sales Class", use_container_width=True)

        if submitted:
            # Create a DataFrame from the user's input
            single_item_data = pd.DataFrame([{
                'Item_Weight': item_weight, 'Item_Fat_Content': item_fat_content, 'Item_Visibility': item_visibility,
                'Item_Type': item_type, 'Item_MRP': item_mrp, 'Outlet_Establishment_Year': outlet_establishment_year,
                'Outlet_Size': outlet_size, 'Outlet_Location_Type': outlet_location_type, 'Outlet_Type': outlet_type
            }])

            # Get the prediction
            with st.spinner("🤖 Analyzing..."):
                prediction = predict(single_item_data, selected_model, scaler, model_columns, imputation_values)
                sales_class = prediction[0]

            st.markdown("---")
            st.subheader("📈 Prediction Result")
            if sales_class == "High":
                st.success(f"**Result: High Sales** - This product is predicted to be a top seller!", icon="🏆")
            elif sales_class == "Medium":
                st.info(f"**Result: Medium Sales** - This product is predicted to have average sales performance.", icon="📊")
            else:
                st.warning(f"**Result: Low Sales** - This product is predicted to be a slow-moving item.", icon="📉")


elif page == "📂 Batch Prediction":
    st.header("🗂️ Upload a File for Batch Prediction")
    st.markdown(f"You are currently using the **`{selected_model_name}`** model.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview of Uploaded Data:")
            st.dataframe(batch_df.head())

            if st.button("🔮 Run Batch Prediction", use_container_width=True, type="primary"):
                with st.spinner("Processing your file... This may take a moment."):
                    predictions = predict(batch_df, selected_model, scaler, model_columns, imputation_values)
                    result_df = batch_df.copy()
                    result_df['Predicted_Sales_Class'] = predictions

                st.subheader("✅ Batch Prediction Results")
                st.dataframe(result_df)

                # Provide a download button for the results
                csv_results = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv_results,
                    file_name=f'prediction_results_{selected_model_name}.csv',
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
