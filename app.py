from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('FINAL1.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    features = [
        float(request.form['gender']),
        float(request.form['10th_percentage']),
        float(request.form['medium_of_education']),
        float(request.form['hsc_or_diploma']),
        float(request.form['board_for_12th_exam']),
        float(request.form['12th_percentage']),
        float(request.form['semester_1_cgpa']),
        float(request.form['semester_2_cgpa']),
        float(request.form['semester_3_sgpa']),
        float(request.form['semester_4_cgpa']),
        float(request.form['semester_5_cgpa'])
    ]

    # Make prediction
    input_data = pd.DataFrame([features], columns=['Gender', '10th Percentage', 'Medium of School Education', 
                                                  'HSC / Diploma', 'Name of Board for 12th Exam', 
                                                  '12th Percentage', 'Semester 1 CGPA', 'Semester 2 CGPA', 
                                                  'Semester 3 SGPA', 'Semester 4 CGPA', 'Semester 5 CGPA'])
    prediction = model.predict(input_data)

    # Prepare data for result display
    result_text = "Placed" if prediction[0] == 1 else "Not Placed"
    return render_template('result.html', result=result_text)

if __name__ == '__main__':
    app.run(debug=True)
