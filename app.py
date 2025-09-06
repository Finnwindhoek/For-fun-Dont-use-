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
    Returns:
        A tuple containing the models dict, scaler, and model columns.
    """
    try:
        assets = joblib.load("sales_classifier.joblib")
        return assets['models'], assets['scaler'], assets['columns']
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None

# Load the assets into global variables.
models, scaler, model_columns = load_assets()   


# ----------------------------
# Prediction Function
# ----------------------------
def predict(data: pd.DataFrame, model, scaler, model_cols):
    df = data.copy()
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'Non-Edible': 'Low Fat'})
    df_encoded = pd.get_dummies(df)
    final_df = df_encoded.reindex(columns=model_cols, fill_value=0)
    scaled_data = scaler.transform(final_df)
    return model.predict(scaled_data)


# ----------------------------
# Main User Interface
# ----------------------------
st.title("🛒 Sales Predictor Pro")
st.markdown("A tool to predict grocery sales performance. You can make a single prediction or upload a CSV file for batch predictions.")

# Check if the assets were loaded successfully before building the rest of the UI.
if models is None or scaler is None or model_columns is None:
    st.warning("Application cannot start because the model assets failed to load.")
else:
    # --- Sidebar: Model Selection ---
    st.sidebar.header("⚙️ Model Settings")
    model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))
    chosen_model = models[model_choice]

    # --- Sidebar Inputs (same as before) ---
    st.sidebar.subheader("📊 Single Prediction Inputs")
    item_weight = st.sidebar.number_input("Item Weight (kg)", min_value=0.1, value=12.5, step=0.1)
    item_fat_content = st.sidebar.selectbox("Item Fat Content", ['Low Fat', 'Regular', 'Non-Edible'])
    item_visibility = st.sidebar.slider("Item Visibility", min_value=0.0, max_value=1.0, value=0.05, step=0.001)
    item_mrp = st.sidebar.number_input("Item MRP (Price)", min_value=10.0, value=140.0, step=0.5)
    item_type = st.sidebar.selectbox("Item Type", ['Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy', 'Canned', 'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 'Meat', 'Breads', 'Hard Drinks', 'Others', 'Starchy Foods', 'Breakfast', 'Seafood'])
    outlet_establishment_year = st.sidebar.slider("Outlet Establishment Year", 1985, 2025, 2002)
    outlet_size = st.sidebar.selectbox("Outlet Size", ['Medium', 'Small', 'High'])
    outlet_location_type = st.sidebar.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
    outlet_type = st.sidebar.selectbox("Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

 

    # Create tabs for the main page content.
    tab1, tab2 = st.tabs([f"📊 Single Prediction ({model_choice})", f"📂 Batch Prediction ({model_choice})"])

    # ----------------------------
    # Tab 1: Single Prediction UI
    # ----------------------------
    with tab1:
        if st.button("🚀 Predict Sales Class", type="primary", use_container_width=True):
            input_data = {
                'Item_Weight': item_weight, 'Item_Fat_Content': item_fat_content, 'Item_Visibility': item_visibility,
                'Item_Type': item_type, 'Item_MRP': item_mrp, 'Outlet_Establishment_Year': outlet_establishment_year,
                'Outlet_Size': outlet_size, 'Outlet_Location_Type': outlet_location_type, 'Outlet_Type': outlet_type
            }
            input_df = pd.DataFrame([input_data])
            prediction = predict(input_df, chosen_model, scaler, model_columns)
            
            st.subheader(f"Prediction Result ({model_choice})")
            sales_class = prediction[0]
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
                            predictions = predict(batch_df, chosen_model, scaler, model_columns)
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

