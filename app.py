from flask import Flask, request, render_template, send_file
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

model = joblib.load('decision_tree_model.joblib')
scaler = StandardScaler()

data = pd.read_csv('Credit.csv')
X = data[['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education']]
y = data['Balance']

X_train_scaled = scaler.fit_transform(X)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train_scaled, y)

threshold = 700

def process_csv(file):
    try:
        data = pd.read_csv(file)
        X = data[['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education']]
        X_scaled = scaler.transform(X)
        data['Predicted_Balance'] = model.predict(X_scaled)
        data['Status'] = data['Predicted_Balance'].apply(lambda x: 'Approved' if x >= 700 else 'Denied')
        return data
    except Exception as e:
        return str(e)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        income = float(request.form['income'])
        limit = float(request.form['limit'])
        rating = float(request.form['rating'])
        cards = float(request.form['cards'])
        age = float(request.form['age'])
        education = float(request.form['education'])

        input_data = pd.DataFrame([[income, limit, rating, cards, age, education]],
                                  columns=['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education'])
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)

        if prediction[0] >= 700:
            decision = "Approved"
            reason = "The predicted balance is sufficient for loan approval."
        else:
            decision = "Denied"
            reason = "The predicted balance is below the threshold for loan approval."

        return render_template("index.html", prediction=f"Predicted Balance: ${prediction[0]:.2f}",
                               decision=decision, reason=reason)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

@app.route("/upload", methods=['POST'])
def upload():
    try:
        file = request.files['file']
        result = process_csv(file)
        if isinstance(result, pd.DataFrame):
            result.to_csv('result.csv', index=False)
            return send_file('result.csv', as_attachment=True)
        else:
            return render_template("index.html", prediction=f"Error: {result}")
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
