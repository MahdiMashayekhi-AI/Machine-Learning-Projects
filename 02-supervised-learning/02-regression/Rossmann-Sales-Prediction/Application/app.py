import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        DayOfWeek = int(float(request.form.get('DayOfWeek')))
        Customers = int(float(request.form.get('Customers')))
        Open = int(float(request.form.get('Open')))
        Promo = int(float(request.form.get('Promo')))
        StateHoliday = int(float(request.form.get('StateHoliday')))
        SchoolHoliday = int(float(request.form.get('SchoolHoliday')))
        Day = int(float(request.form.get('Day')))
        Month = int(float(request.form.get('Month')))

        data = pd.DataFrame({
            'DayOfWeek': [DayOfWeek],
            'Customers': [Customers],
            'Open': [Open],
            'Promo': [Promo],
            'StateHoliday': [StateHoliday],
            'SchoolHoliday': [SchoolHoliday],
            'Day': [Day],
            'Month': [Month],
        })

        prediction = model.predict(data)[0]
        # response = f'Predicted sales: {prediction}'

        return render_template('index.html', result=prediction)
    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(debug=True)
