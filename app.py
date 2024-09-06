from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_file_path = 'random_forest_model.pkl'
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

# Load the label encoders (if any were used)
label_encoders = {}  # Load or create this if you used LabelEncoders

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extracting input values from the form
    input_values = [float(x) for x in request.form.values()]
    
    # Convert input to DataFrame (assuming input order matches the model's training order)
    input_df = pd.DataFrame([input_values], columns=['step', 'customer', 'age', 'gender', 'zipcodeOri', 'merchant',
                                                     'zipMerchant', 'category', 'amount'])
    
    # Apply any necessary label encoding (if applicable)
    #for column in label_encoders:
    #    input_df[column] = label_encoders[column].transform(input_df[column])

    # Predict the output
    prediction = model.predict(input_df)[0]
    
    # Map prediction to label and assign a CSS class
    if prediction == 1:
        output = "Fraudulent"
        result_class = "fraud"
    else:
        output = "Not Fraudulent"
        result_class = "not-fraud"

    return render_template('index.html', prediction_text=f'The transaction is {output}.', result_class=result_class)

if __name__ == "__main__":
    app.run(debug=True)
