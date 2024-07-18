from flask import Flask, request, render_template
import pickle
import pandas as pd

# Load the trained model
with open('RF.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about", methods=["POST", "GET"])
def about():
    return render_template("about.html")

@app.route("/input")
def input():
    return render_template("second.html")

@app.route('/submit', methods=["POST"])  # Specify POST method
def submit():
    # Reading input values from the form
    input_feature = [float(x) for x in request.form.values()]
    names = ['i', 'z', 'modelFlux_z', 'petroRad_g', 'petroRad_r', 'petroFlux_z', 'petroR50_u', 'petroR50_g', 'petroR50_i', 'petroR50_r']
    
    print("Number of columns in names:", len(names))
    print("Number of columns in input_feature:", len(input_feature))
    print("Column names:", names)
    
    data = pd.DataFrame([input_feature], columns=names)
    
    # Make prediction
    prediction = model.predict(data)
    
    # Render the output template with the prediction result
    if prediction[0] == 0:
        print(prediction)
        return render_template('final.html', prediction='starforming')
    else:
        return render_template('final.html', prediction='starbursting')

if __name__ == "__main__":
    app.run(debug=True)
