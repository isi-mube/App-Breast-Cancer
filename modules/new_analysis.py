import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Load data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnose'] = data.target
df.columns = [column.lower().replace(" ", "_") for column in df.columns]

# Selected features for the model
selected_features_model = ['worst_concave_points', 'worst_perimeter', 'smoothness_error', 'worst_area', 'mean_concave_points']

# Split data for training and testing
X = df[selected_features_model]
y = df['diagnose']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Retrain the model with scaled data
model.fit(X_train_scaled, y_train)

# Cytological terms mapping for user input
cytological_terms = {
    'worst_concave_points': 'Marked nuclear indentation',
    'worst_perimeter': 'Irregular nuclear membrane',
    'smoothness_error': 'Variability in nuclear membrane smoothness',
    'worst_area': 'Increased nuclear area',
    'mean_concave_points': 'Nuclear indentations'
}

def new_analysis():
    st.header('Breast Cell Morphometrics Analysis')
    st.write("Please enter the values for the following nucei features:")

    with st.form(key='my_form'):
        input_data = {}
        for feature in selected_features_model:
            min_value = float(df[feature].min())
            max_value = float(df[feature].max())
            mean_value = float(df[feature].mean())
            input_data[feature] = st.slider(
                f"{cytological_terms.get(feature, feature)}", 
                min_value=min_value, max_value=max_value, value=mean_value
            )
        
        submit_button = st.form_submit_button("Results")
        
    if submit_button:
        # Scale the input data
        input_values = [input_data[feature] for feature in selected_features_model]
        input_scaled = scaler.transform([input_values])

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)
        diagnosis = 'Benign' if prediction[0] == 1 else 'Malignant'
        certainty = prediction_proba[0][prediction[0]] * 100
        
        if diagnosis == 'Malignant':
            st.error(f'The predicted diagnosis is: **{diagnosis}** with a certainty of {certainty:.2f}%')
            st.image('images/malign.png')
        else:
            st.success(f'The predicted diagnosis is: **{diagnosis}** with a certainty of {certainty:.2f}%')
            st.image('images/benign.png')