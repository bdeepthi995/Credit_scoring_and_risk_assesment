from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)


data = pd.read_csv('Credit.csv')

X = data[['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education']]
y = data['Balance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = DecisionTreeRegressor(random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'decision_tree_model.joblib')
threshold = 700


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

        if prediction[0] >= threshold:
            decision = "Approved"
            reason = "The predicted balance is sufficient for loan approval."
        else:
            decision = "Denied"
            reason = "The predicted balance is below the threshold for loan approval."

        return render_template("index.html", prediction=f"Predicted Balance: ${prediction[0]:.2f}",
                               decision=decision, reason=reason,
                               loan_decision=decision, loan_reason=reason)
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
