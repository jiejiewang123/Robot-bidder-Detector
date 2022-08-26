

import numpy as np
from flask import Flask, request, render_template
import pickle

# create an app object using the Flask class
app = Flask(__name__)

# load the trained model
model = pickle.load(open('models/rf_model.pkl', 'rb'))

# define the route to be home
# use the route decorator to tell Flask what URL should trigger our function

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features_input = [float(x) for x in request.form.values()]
    features = [np.array(features_input)]
    prediction = model.predict(features) # features must be like [[11,2,..]]

    output = prediction[0]
    if output == 0:
        re = 'robot'
    else:
        re = 'human'

    return render_template('index.html', prediction_result = "This bidder is {}".format(re))


if __name__ == "__main__":
    app.run(debug = True, host = "127.0.0.1", port = 5000)
