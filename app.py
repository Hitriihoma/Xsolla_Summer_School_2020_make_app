from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import pickle
import gunicorn



from flask import Flask
from flask import request
import requests
from flask import jsonify

import os
import json
from ast import literal_eval
import traceback

app = Flask(__name__)


#load models from file
vec = pickle.load(open("./models/tfidf.pickle", "rb"))
model = pickle.load(open("./models/MLPClassifier_model.pickle", "rb"))


# test output
@app.route("/")  
def hello():
    resp = {'message':"Hello World!"}
    
    response = jsonify(resp)
    
    return response

'''
prediction of category
Input: {"user_message":"example123rfssg gsfgfd"}
Output: {
    "category": -1,
    "message": "ok",
    "prediciton": [
        [
            0.04927693250080478,
            0.44692604628460875,
            0.5037970212145865
        ]
    ]
}
''' 
@app.route("/categoryPrediction" , methods=['GET', 'POST'])  
def registration():
    resp = {'message':'ok'
           ,'category': -1
           }

    try:
        getData = request.get_data()
        json_params = json.loads(getData) 
        
        #make prediction. return in parameter 'prediction'
        prediction = model.predict_proba(vec.transform([json_params['user_message']]).toarray()).tolist()
        resp['prediciton'] = prediction
        
    except Exception as e: 
        print(e)
        resp['message'] = e
      
    response = jsonify(resp)
    
    return response

        

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(debug=False, port=port, host='0.0.0.0' , threaded=True)



