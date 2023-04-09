from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('trained_model.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return "Hello world"

@app.route('/predict', methods = ['POST'])
def predict():
    age = request.form.get('age')
    gender = request.form.get('gender')
    type_of_room = request.form.get('type_of_room')
    smoke = request.form.get('smoke')
    study_time = request.form.get('study_time')
    family_income = request.form.get('family_income')

    input_query = np.array([[age,gender,type_of_room,smoke,study_time,family_income]])

    result = model.predict(input_query)[0]

    return jsonify({'hostel_name':str(result)})


if __name__ == '__main__':
    app.run(debug=True)