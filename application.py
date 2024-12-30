from flask import Flask, render_template, request
import pandas as pd
import pickle
import sklearn

# Initialize Flask app
app = Flask(__name__)

# Load dataset and model
bmd = pd.read_csv('cleaned_dataset.csv')
with open('ml_pipeline.pkl', 'rb') as f:
    loaded_pipeline = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Define feature names
feature_names = [
    'Gender', 'Age', 'BMI', 'FN', 'FNT', 'TL', 'TLT',
    'Ca', 'Calcitriol', 'OP', 'Fracture', 'Smoking', 'Drinking'
]


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probabilities = None

    if request.method == 'POST':
        # Collect form data
        form_data = request.form

        # Extract input values in the correct order
        input_data = [[
            int(form_data['Gender']),
            float(form_data['Age']),
            float(form_data['BMI']),
            float(form_data['FN']),
            float(form_data['FNT']),
            float(form_data['TL']),
            float(form_data['TLT']),
            float(form_data['Ca']),
            int(form_data['Calcitriol']),
            int(form_data['OP']),
            int(form_data['Fracture']),
            int(form_data['Smoking']),
            int(form_data['Drinking']),
        ]]

        # Create a DataFrame for prediction
        input_df = pd.DataFrame(input_data, columns=feature_names)

        # Predict using the pipeline
        prediction = loaded_pipeline.predict(input_df)
        predicted_label = le.inverse_transform(prediction)[0]

        # Get prediction probabilities
        probabilities = loaded_pipeline.predict_proba(input_df)[0]

        return render_template(
            'index.html',
            Gender=sorted(bmd['Gender'].unique()),
            Age=sorted(bmd['Age'].unique()),
            BMI=sorted(bmd['BMI'].unique()),
            FN=sorted(bmd['FN'].unique()),
            FNT=sorted(bmd['FNT'].unique()),
            TL=sorted(bmd['TL'].unique()),
            TLT=sorted(bmd['TLT'].unique()),
            Ca=sorted(bmd['Ca'].unique()),
            Calcitriol=sorted(bmd['Calcitriol'].unique()),
            OP=sorted(bmd['OP'].unique()),
            Fracture=sorted(bmd['Fracture'].unique()),
            Smoking=sorted(bmd['Smoking'].unique()),
            Drinking=sorted(bmd['Drinking'].unique()),
            predicted_label=predicted_label,
            probabilities=probabilities
        )

    return render_template(
        'index.html',
        Gender=sorted(bmd['Gender'].unique()),
        Age=sorted(bmd['Age'].unique()),
        BMI=sorted(bmd['BMI'].unique()),
        FN=sorted(bmd['FN'].unique()),
        FNT=sorted(bmd['FNT'].unique()),
        TL=sorted(bmd['TL'].unique()),
        TLT=sorted(bmd['TLT'].unique()),
        Ca=sorted(bmd['Ca'].unique()),
        Calcitriol=sorted(bmd['Calcitriol'].unique()),
        OP=sorted(bmd['OP'].unique()),
        Fracture=sorted(bmd['Fracture'].unique()),
        Smoking=sorted(bmd['Smoking'].unique()),
        Drinking=sorted(bmd['Drinking'].unique())
    )


if __name__ == "__main__":
    app.run(debug=True)
