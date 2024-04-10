 # Prepare data for result display
    result_text = "Placed" if prediction[0] == 1 else "Not Placed"
    return render_template('result.html', result=result_text)