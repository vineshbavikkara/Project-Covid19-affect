#  For Deployment
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    response=[x for x in request.form.values()]
    response = pd.DataFrame(response).T
    data = pd.read_csv('Albania_Cleaned_Data.csv')
    data = data.drop(columns=['Sales_Performance'])
    response.columns = ['Industry_Type','Current_Situation','Direct_Export','Indirect_Export','Hours_Worked_Per_Week','Demand_For_Product','Supply_Of_Raw_Materials','Financial_Performance','Online_Business']
    response = pd.concat([response,data],axis=0,ignore_index=True)
    # Encoding of Label features
    response = pd.get_dummies(response)
    response = response.astype(int)
    response = pd.DataFrame(response.iloc[:1,:])
    pred = model.predict(response)

    if pred[0]==1:
        return render_template("index.html",predicted="Covid-19 Does not Affect the Performance of Your Company.")
    return render_template("index.html",predicted="Covid-19 Affected the Performance of Your Company.")

if __name__=="__main__":
    app.run(debug=True)
    
