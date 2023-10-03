import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression

# Initialize the LabelEncoder
le = LabelEncoder()

# Load your trained model (replace 'iris_classifier_model.pkl' with the correct file path)
model_file_path = 'iris_classifier_model.pkl'
model = joblib.load(model_file_path)

# Load the dataset for preprocessing (replace 'Cleaned Iris.csv' with your dataset file)
df = pd.read_csv('Cleaned Iris.csv')

# Apply LabelEncoder to the 'Species' column
df['Species'] = le.fit_transform(df['Species'])

# Separate features (X) and target (y)
X = df.drop(columns=['Species'])
y = df['Species']

# Create a ColumnTransformer with StandardScaler applied to selected columns
columns_to_scale = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
column_transformer = make_column_transformer(
    (StandardScaler(), columns_to_scale),
    remainder='drop'  # Drop any columns not specified in transformers
)

# Fit the column_transformer with the training data
X_transformed = column_transformer.fit_transform(X)

# logistic regression
lr = LogisticRegression()

# Create a Pipeline that fits a Logistic Regression model
pipe = make_pipeline(column_transformer, lr)

# Fit the model with the transformed training data
pipe.fit(X, y)

# Define a function to predict the species
def predict_species(input_data):
    # Make predictions using the loaded model
    y_pred = pipe.predict(input_data)

    # Decode the label using the inverse transform of LabelEncoder
    predicted_species = le.inverse_transform(y_pred)

    return predicted_species

# Create the Streamlit app
st.title('Iris Species Classifier')

sepal_length = st.number_input('Enter Sepal Length (cm): ')
sepal_width = st.number_input('Enter Sepal Width (cm): ')
petal_length = st.number_input('Enter Petal Length (cm): ')
petal_width = st.number_input('Enter Petal Width (cm): ')

user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button('Predict'):
    predicted_species = predict_species(user_input)
    st.write(f'Predicted Iris Species: {predicted_species[0]}')
