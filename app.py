from flask import Flask, render_template,request,url_for
import joblib
import numpy as np

app=Flask(__name__)
model=joblib.load("placement_predictor.joblib")

@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict_placement():
    cgpa=float(request.form.get("cgpa"))
    iq=int(request.form.get("iq"))
    profile_score=int(request.form.get("profile_score"))

    user_input=np.array([[cgpa,iq,profile_score]])
    result=model.predict(np.array([cgpa,iq,profile_score]).reshape(1,3))

    if result[0]==1:
        result="Placed"

    else:
        result="Not Placed"    

    return render_template("index.html",result=result)


if __name__=="__main__":
    app.run(debug=True)

