from flask import Flask, render_template, request
import model as m 

app = Flask(__name__)

@app.route("/",methods=["POST","GET"])
def marks():
    mk=0
    if request.method == "POST":
        hrs = request.form["hrs"]
        marks_pred = m.marks_prediction(hrs)
        mk=marks_pred

    return render_template("index.html",my_marks=mk)

if __name__ == "__main__":
    app.run(debug=True)

