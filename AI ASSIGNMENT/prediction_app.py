import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

# Load the dataset
df = pd.read_csv('Cleaned_Data_RSL.csv')

# Split data into features (X) and target variable (y)
X = df.drop('Value', axis=1)
y = df['Value']

# Preprocessing
categorical_features = ["GOV", "Level of government", "TAX", "Revenue category"]
one_hot = OneHotEncoder()
transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")
transformed_X = transformer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2, random_state=42)

# Load the trained model
model = joblib.load('random_forest_model99.joblib')

# Streamlit app
def main():
    st.title('Revenue Prediction')

    # Input fields
    govs = df['GOV'].unique()
    selected_gov = st.selectbox('Select Government Level', govs)

    levels = df['Level of government'].unique()
    selected_level = st.selectbox('Select Level of Government', levels)

    taxes = df['TAX'].unique()
    selected_tax = st.selectbox('Select Tax Category', taxes)

    categories = df['Revenue category'].unique()
    selected_category = st.selectbox('Select Revenue Category', categories)

    # year = st.number_input('Enter Year', min_value=int(df['Year'].min()), max_value=int(df['Year'].max()))
    year = st.number_input('Enter Year', min_value=int(df['Year'].min()), max_value=2030)

    # Preprocess input data
    input_data = pd.DataFrame({'GOV': [selected_gov], 'Level of government': [selected_level],
                               'TAX': [selected_tax], 'Revenue category': [selected_category],
                               'Year': [year]})
    input_data_transformed = transformer.transform(input_data)

    # Predict button
    if st.button('Predict'):
        prediction = model.predict(input_data_transformed)
        st.success(f'Predicted Value: {prediction[0]:.3f}')

if __name__ == '__main__':
    main()
