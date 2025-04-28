from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open('loan_approval_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
           
            no_of_dependents = float(request.form['no_of_dependents'])
            education = int(request.form['education'])
            self_employed = int(request.form['self_employed'])
            income_annum = float(request.form['income_annum'])
            loan_amount = float(request.form['loan_amount'])
            loan_term = float(request.form['loan_term'])
            cibil_score = float(request.form['cibil_score'])
            residential_assets_value = float(request.form['residential_assets_value'])
            commercial_assets_value = float(request.form['commercial_assets_value'])
            luxury_assets_value = float(request.form['luxury_assets_value'])
            bank_asset_value = float(request.form['bank_asset_value'])

        
            new_data = pd.DataFrame({
                'no_of_dependents': [no_of_dependents],
                'education': [education],
                'self_employed': [self_employed],
                'income_annum': [income_annum],
                'loan_amount': [loan_amount],
                'loan_term': [loan_term],
                'cibil_score': [cibil_score],
                'residential_assets_value': [residential_assets_value],
                'commercial_assets_value': [commercial_assets_value],
                'luxury_assets_value': [luxury_assets_value],
                'bank_asset_value': [bank_asset_value]
            })

            prediction = model.predict(new_data)

            result = " KREDİ ONAYLANDI!" if prediction[0] == 1 else " KREDİ REDDEDİLDİ."

            return render_template('index.html', prediction_text=result)

        except Exception as e:
            return render_template('index.html', prediction_text=f"Bir hata oluştu: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
