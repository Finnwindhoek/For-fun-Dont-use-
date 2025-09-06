import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page Configuration
# ----------------------------
# Sets the browser tab title, icon, and layout for the Streamlit page.
st.set_page_config(
    page_title="Sales Predictor Pro",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ----------------------------
# Load Assets
# ----------------------------
# This function loads the model, scaler, and column names from the single .joblib file.
# @st.cache_resource ensures this expensive operation runs only once, making the app faster.
@st.cache_resource
def load_assets():
    """
    Loads all necessary assets from a single joblib file.
    This includes the trained model, the feature scaler, and the list of model column names.
    Returns:
        A tuple containing the model, scaler, and model columns.
    """
    try:
        assets = joblib.load("sales_classifier.joblib")
        return assets['model'], assets['scaler'], assets['columns']
    except FileNotFoundError:
        st.error("Asset file 'sales_classifier.joblib' not found. Please ensure it's in the same directory as the app.")
        return None, None, None
    except KeyError:
        st.error("The asset file is corrupted or invalid. Please recreate it and ensure it contains 'model', 'scaler', and 'columns' keys.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the asset file: {e}")
        return None, None, None

# Load the assets into global variables.
model, scaler, model_columns = load_assets()


# ----------------------------
# Prediction Function
# ----------------------------
def predict(data: pd.DataFrame, model, scaler, model_cols):
    """
    Preprocesses the input data and returns model predictions.
    Args:
        data (pd.DataFrame): The input data for prediction.
        model: The trained machine learning model.
        scaler: The fitted StandardScaler instance.
        model_cols (list): The exact list of column names the model was trained on.
    Returns:
        An array of predictions.
    """
    df = data.copy()
    
    # Step 1: Standardize 'Item_Fat_Content' values.
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'Non-Edible': 'Low Fat'})
    
    # Step 2: Perform one-hot encoding on categorical features.
    df_encoded = pd.get_dummies(df)
    
    # Step 3: Align columns with the model's training columns. This is crucial.
    # It adds any missing columns (with a value of 0) and removes any extra columns.
    final_df = df_encoded.reindex(columns=model_cols, fill_value=0)
    
    # Step 4: Scale the features using the loaded scaler.
    scaled_data = scaler.transform(final_df)
    
    # Step 5: Make predictions.
    predictions = model.predict(scaled_data)
    
    return predictions


# ----------------------------
# Main User Interface
# ----------------------------
st.title("🛒 Sales Predictor Pro")
st.markdown("A tool to predict grocery sales performance. You can make a single prediction or upload a CSV file for batch predictions.")

# Check if the assets were loaded successfully before building the rest of the UI.
if model is None or scaler is None or model_columns is None:
    st.warning("Application cannot start because the model assets failed to load. Please check the error messages above.")
else:
    # --- Sidebar for Single Prediction Inputs ---
    st.sidebar.header("📊 Single Prediction Inputs")
    
    st.sidebar.subheader("Product Details")
    item_weight = st.sidebar.number_input("Item Weight (kg)", min_value=0.1, value=12.5, step=0.1, help="Enter the weight of the product in kilograms.")
    item_fat_content = st.sidebar.selectbox("Item Fat Content", ['Low Fat', 'Regular', 'Non-Edible'], key='single_fat')
    item_visibility = st.sidebar.slider("Item Visibility", min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.3f", help="The percentage of total display area for the product.")
    item_mrp = st.sidebar.number_input("Item MRP (Price)", min_value=10.0, value=140.0, step=0.5, help="Maximum Retail Price of the product.")
    item_type = st.sidebar.selectbox("Item Type", ['Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks', 'Others', 'Starchy Foods', 'Breakfast', 'Seafood'], key='single_type')

    st.sidebar.subheader("Outlet Details")
    outlet_establishment_year = st.sidebar.slider("Outlet Establishment Year", 1985, 2025, 2002)
    outlet_size = st.sidebar.selectbox("Outlet Size", ['Medium', 'Small', 'High'], key='single_size')
    outlet_location_type = st.sidebar.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'], key='single_location')
    outlet_type = st.sidebar.selectbox("Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'], key='single_outlet')


    # Create tabs for the main page content.
    tab1, tab2 = st.tabs(["📊 Single Prediction", "📂 Batch Prediction"])

    # ----------------------------
    # Tab 1: Single Prediction UI
    # ----------------------------
    with tab1:
        st.header("Make a Single Prediction")
        st.write("Use the input fields in the sidebar to enter the details for a product.")
        
        # Prediction button and logic.
        if st.button("🚀 Predict Sales Class", type="primary", use_container_width=True):
            input_data = {
                'Item_Weight': item_weight, 'Item_Fat_Content': item_fat_content, 'Item_Visibility': item_visibility,
                'Item_Type': item_type, 'Item_MRP': item_mrp, 'Outlet_Establishment_Year': outlet_establishment_year,
                'Outlet_Size': outlet_size, 'Outlet_Location_Type': outlet_location_type, 'Outlet_Type': outlet_type
            }
            input_df = pd.DataFrame([input_data])
            prediction = predict(input_df, model, scaler, model_columns)
            
            st.subheader("Prediction Result")
            sales_class = prediction[0]
            
            # Display the result using a metric for a clean look.
            st.metric(label="Predicted Sales Class", value=sales_class)
            
            if sales_class == "High":
                st.success("This product is predicted to be a **top seller**.")
            elif sales_class == "Medium":
                st.info("This product is predicted to have **average sales performance**.")
            else:
                st.warning("This product is predicted to be a **slow-moving item**.")

    # ----------------------------
    # Tab 2: Batch Prediction UI
    # ----------------------------
    with tab2:
        st.header("Upload a CSV File for Batch Prediction")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.dataframe(batch_df.head())

                if st.button("🔮 Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner('Processing your file... This may take a moment.'):
                        # Ensure the uploaded data has the required columns before prediction.
                        required_cols = ['Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
                        
                        if not all(col in batch_df.columns for col in required_cols):
                            st.error(f"Error: The uploaded CSV must contain the following columns: {', '.join(required_cols)}")
                        else:
                            predictions = predict(batch_df[required_cols], model, scaler, model_columns)
                            result_df = batch_df.copy()
                            result_df['Predicted_Sales_Class'] = predictions
                            
                            st.subheader("Batch Prediction Results")
                            st.dataframe(result_df)

                            # Provide a download button for the results.
                            csv_results = result_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_results,
                                file_name='prediction_results.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")

# --- End of App ---

