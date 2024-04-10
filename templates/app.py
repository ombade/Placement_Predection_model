import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open('FINAL1.pkl', 'rb') as file:
    model = pickle.load(file)

def predict(features):
    # Make prediction
    input_data = pd.DataFrame([features], columns=['Gender', '10th Percentage', 'Medium of School Education', 
                                                  'HSC / Diploma', 'Name of Board for 12th Exam', 
                                                  '12th Percentage', 'Semester 1 CGPA', 'Semester 2 CGPA', 
                                                  'Semester 3 SGPA', 'Semester 4 CGPA', 'Semester 5 CGPA'])
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    st.title('Placement Prediction')

    # Form inputs
    gender = st.selectbox('Gender', ['Male', 'Female'])
    tenth_percentage = st.number_input('10th Percentage')
    medium_of_education = st.selectbox('Medium of Education', ['English', 'Hindi', 'Other'])
    hsc_or_diploma = st.selectbox('HSC / Diploma', ['HSC', 'Diploma'])
    board_for_12th_exam = st.selectbox('Name of Board for 12th Exam', ['State Board', 'CBSE', 'ICSE', 'Others'])
    twelfth_percentage = st.number_input('12th Percentage')
    semester_1_cgpa = st.number_input('Semester 1 CGPA')
    semester_2_cgpa = st.number_input('Semester 2 CGPA')
    semester_3_sgpa = st.number_input('Semester 3 SGPA')
    semester_4_cgpa = st.number_input('Semester 4 CGPA')
    semester_5_cgpa = st.number_input('Semester 5 CGPA')

    # One-hot encode categorical variables
    gender_encoded = 1 if gender == 'Male' else 0
    medium_of_education_encoded = 0
    if medium_of_education == 'Hindi':
        medium_of_education_encoded = 1
    elif medium_of_education == 'Other':
        medium_of_education_encoded = 2
    hsc_or_diploma_encoded = 1 if hsc_or_diploma == 'HSC' else 0
    board_for_12th_exam_encoded = ['State Board', 'CBSE', 'ICSE', 'Others'].index(board_for_12th_exam)

    # Make prediction on button click
    if st.button('Predict'):
        features = [gender_encoded, tenth_percentage, medium_of_education_encoded, hsc_or_diploma_encoded,
                    board_for_12th_exam_encoded, twelfth_percentage, semester_1_cgpa, semester_2_cgpa,
                    semester_3_sgpa, semester_4_cgpa, semester_5_cgpa]

        prediction = predict(features)
        result_text = "Placed" if prediction == 1 else "Not Placed"
                # Set text color based on prediction result
        text_color = "black"   
        if result_text == "Placed":
            text_color = "green"
        else:
            text_color = "red"
        
    st.markdown(
        f"""
        <div >
            <h2>Prediction Result</h2>
            <div style="margin-top: 30px; font-style: italic;">
                <h2 style="color: {text_color};">{"The Student has high chances of Placement" if result_text == "Placed" else "The Student has Less chances of Placement"}</h2>
                <h4>"Success is not final, failure is not fatal: It is the courage to continue that counts." - Winston Churchill</h4>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
if __name__ == '__main__':
    main()
