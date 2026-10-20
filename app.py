import pickle
import pandas as pd
import streamlit as st
import tensorflow as tf

## page setting

st.set_page_config(
    page_title="ANN Salary Prediction",
    page_icon=":bar_chart:",
    layout="wide"
)


## load the trained model and processing files

try:
    model = tf.keras.models.load_model("model.h5")

    with open("label_encoder_gender.pkl", "rb") as file:
        label_encoder_gender = pickle.load(file)

    with open("onehot_encoder_geo.pkl", "rb") as file:
        onehot_encoder_geo = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

except Exception as e:
    st.error(f"Error loading model or processing files: {e}")
    st.stop()


## App title and introduction

st.title("Bank Customer Salary Prediction")
st.write(
    "Fill in the customer profile and press "
    "**Predict Customer Salary** to get the prediction."
)

st.success("ANN Model and preprocessing files loaded successfully.")


# organize the aapp using tabs

prediction_tab, learning_tab = st.tabs(["Predict Salary", "How does it work?"])


with prediction_tab:
    st.subheader("Customer Information")
    ## create the input form.

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
            gender = st.selectbox("Gender", label_encoder_gender.classes_)
            credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, value=650, step=1)

        with col2:
            age = st.number_input("Age", min_value=18, max_value=92, value=40)
            tenure = st.slider("Tenure (Years)", min_value=0, max_value=10, value=5)
            num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)

        with col3:
            has_card_text = st.selectbox("Has Credit Card", ["Yes", "No"])
            balance = st.number_input("Balance", min_value=0.0, value=0.0, step=1000.0)
            exited = st.selectbox("Has Exited", ["Yes", "No"])
            active_text = st.selectbox("Is Active Member", ["Yes", "No"])

        predict_button = st.form_submit_button("Predict Customer Salary", type="primary", use_container_width=True)

    if predict_button:
        has_cr_card = 1 if has_card_text == "Yes" else 0
        is_active_member = 1 if active_text == "Yes" else 0
        is_exited = 1 if exited == "Yes" else 0

        encoded_gender = label_encoder_gender.transform([gender])[0]

        input_data = pd.DataFrame({
            "CreditScore": [credit_score],
            "Gender": [encoded_gender],
            "Age": [age],
            "Tenure": [tenure],
            "Balance": [balance],
            "NumOfProducts": [num_of_products],
            "HasCrCard": [has_cr_card],
            "IsActiveMember": [is_active_member],
            "Exited": [is_exited]
        })

        geography_df = pd.DataFrame({"Geography": [geography]})
        geography_encoded = onehot_encoder_geo.transform(geography_df).toarray()
        geography_df = pd.DataFrame(
            geography_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(["Geography"])
        )

        input_data = pd.concat([input_data, geography_df], axis=1)
        input_data = input_data[scaler.feature_names_in_]

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled, verbose=0)
        predicted_salary = float(prediction[0][0])

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
        "Predicted Estimated Salary",
        f"{predicted_salary:,.2f}"
        )

        
    with st.expander("View Customer Summary"):
        customer_summary = pd.DataFrame(
            {
                "Feature": [
                    "Geography",
                    "Gender",
                    "Credit Score",
                    "Age",
                    "Tenure",
                    "Balance",
                    "Number of Products",
                    "Has Credit Card",
                    "Active Member",
                    "Exited"
                ],
                "Value": [
                    geography,
                    gender,
                    credit_score,
                    age,
                    tenure,
                    balance,
                    num_of_products,
                    has_card_text,
                    active_text,
                    exited

                ]
            }
        )

        st.dataframe(
            customer_summary,
            use_container_width=True,
            hide_index=True
        )
with learning_tab:
    st.subheader("How does salary prediction work?")

    st.write("This app uses deep learning's concept of Artificial Neural Networks(ANN) to predict the estimated salary of a customer based on the features like  age, credit score, balance, geography, and banking activity.")   

    st.markdown("""
    **1. Prepare the inputs**  
    Gender is converted into a number, and geography is converted into
    separate country columns. The saved scaler then scales the inputs using the same rules used during training.

    **2. Predict the salary**  
    The ANN has two hidden layers with 64 and 32 neurons using ReLU
    activation. Its single output neuron uses linear activation to predict a salary amount.

    **3. Evaluate performance**  
    - **MAE:** Average absolute prediction error in salary units.
    - **RMSE:** Measures prediction error with a larger penalty for big mistakes.
    - **R²:** Compares predictions with a mean-salary baseline.
      A negative value means worse performance than that baseline.

    **Classification vs regression**  
    The original churn model predicted whether a customer would leave.
    This regression model predicts a continuous salary amount instead.

    **Limitations**  
    This is a classroom learning exercise. The model showed poor test
    performance, and the available customer details may not explain salary
    well. Predictions should not be used for real banking or financial decisions.
    """)



     
