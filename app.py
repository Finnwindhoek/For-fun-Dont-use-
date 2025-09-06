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
        return assets['models'], assets['scaler'], assets['columns'], assets.get('evaluation', None)
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None, None

# Load the assets into global variables.
models, scaler, model_columns, eval_results = load_assets()


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
st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Predictions"])



# ----------------------------
# Home Page
# ----------------------------
if page == "🏠 Home":
    st.title("🛒 Sales Predictor Pro")
    st.markdown("""
    ### Welcome to **Sales Predictor Pro**
    This application uses **AI models (SVM, KNN, ANN)** to classify products into 
    **High**, **Medium**, or **Low** sales categories.
    
    ---
    #### 🔥 Features:
    - Single prediction from sidebar inputs  
    - Batch prediction from CSV upload  
    - Choose from multiple models: **SVM, KNN, ANN**  
    - Download prediction results  
    """)
    
    if eval_results:
        st.subheader("📈 Model Evaluation Comparison")

        results_df = pd.DataFrame(eval_results)

        # Split into baseline vs tuned
        baseline_models = ["SVM (rbf)", "KNN (k=5)", "ANN (100)"]
        tuned_models = ["SVM (Linear)", "KNN (k=7)", "ANN (2 layers)"]

        baseline_df = results_df[results_df["Model"].isin(baseline_models)]
        tuned_df = results_df[results_df["Model"].isin(tuned_models)]

        col1, col2 = st.columns(2)

        with col1:
            st.write("### 📌 Default Models (Baseline)")
            st.dataframe(baseline_df)
            st.bar_chart(baseline_df.set_index("Model")[["Accuracy", "Macro F1", "Weighted F1"]])

        with col2:
            st.write("### ⚡ Tuned Models (Optimized)")
            st.dataframe(tuned_df)
            st.bar_chart(tuned_df.set_index("Model")[["Accuracy", "Macro F1", "Weighted F1"]])

    st.info("👉 Use the sidebar navigation to switch to the **Predictions** page.")




elif page == "📊 Predictions":
    if models is None or scaler is None or model_columns is None:
        st.warning("❌ Model assets not loaded. Please ensure `sales_classifier.joblib` is available.")
    else:
        st.title("📊 Sales Predictions")
        st.markdown("Choose model, enter product details, and predict sales class.")

        # --- Sidebar: Model Selection ---
        st.sidebar.header("⚙️ Model Settings")
        model_choice = st.sidebar.selectbox("Choose Model", list(models.keys()))
        chosen_model = models[model_choice]

        # --- Sidebar Inputs ---
        st.sidebar.subheader("📊 Single Prediction Inputs")
        item_weight = st.sidebar.number_input("Item Weight (kg)", min_value=0.1, value=12.5, step=0.1)
        item_fat_content = st.sidebar.selectbox("Item Fat Content", ['Low Fat', 'Regular', 'Non-Edible'])
        item_visibility = st.sidebar.slider("Item Visibility", min_value=0.0, max_value=1.0, value=0.05, step=0.001)
        item_mrp = st.sidebar.number_input("Item MRP (Price)", min_value=10.0, value=140.0, step=0.5)
        item_type = st.sidebar.selectbox("Item Type", [
            'Fruits and Vegetables', 'Snack Foods', 'Household', 'Frozen Foods', 'Dairy',
            'Canned', 'Baking Goods', 'Health and Hygiene', 'Soft Drinks', 'Meat',
            'Breads', 'Hard Drinks', 'Others', 'Starchy Foods', 'Breakfast', 'Seafood'
        ])
        outlet_establishment_year = st.sidebar.slider("Outlet Establishment Year", 1985, 2025, 2002)
        outlet_size = st.sidebar.selectbox("Outlet Size", ['Medium', 'Small', 'High'])
        outlet_location_type = st.sidebar.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
        outlet_type = st.sidebar.selectbox("Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

        # --- Tabs for Single/Batch Predictions ---
        tab1, tab2 = st.tabs([f"📊 Single Prediction ({model_choice})", f"📂 Batch Prediction ({model_choice})"])

        # Tab 1: Single Prediction
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
                    st.success("This product is predicted to be a **top seller** ✅")
                elif sales_class == "Medium":
                    st.info("This product is predicted to have **average sales performance** ℹ️")
                else:
                    st.warning("This product is predicted to be a **slow-moving item** ⚠️")

        # Tab 2: Batch Prediction
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
                            required_cols = [
                                'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type',
                                'Item_MRP', 'Outlet_Establishment_Year', 'Outlet_Size',
                                'Outlet_Location_Type', 'Outlet_Type'
                            ]
                            if not all(col in batch_df.columns for col in required_cols):
                                st.error(f"CSV must contain: {', '.join(required_cols)}")
                            else:
                                predictions = predict(batch_df, chosen_model, scaler, model_columns)
                                result_df = batch_df.copy()
                                result_df['Predicted_Sales_Class'] = predictions
                                st.subheader("Batch Prediction Results")
                                st.dataframe(result_df)

                                # Download button
                                csv_results = result_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv_results,
                                    file_name='prediction_results.csv',
                                    mime='text/csv',
                                    use_container_width=True
                                )
                except Exception as e:
                    st.error(f"Error processing file: {e}")

# --- End of App ---

