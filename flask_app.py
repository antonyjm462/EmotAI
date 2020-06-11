from flask import Flask, render_template, request
from decision import Emotion

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    emotion="Enter text to predict"
    confidence="AI"
    emot = Emotion()
    print(emot,flush=True)
    if request.method == 'POST':
        text = request.form['text']
        print(text,flush=True)
        emotion,confidence = emot.test(text)
        print(emotion,confidence)
        return render_template("index.html", emotion=emotion, confidence=confidence)
    else:
        return render_template("index.html", emotion="Enter text to predict" , confidence="AI")


if __name__ == '__main__':
    app.run(debug=True) # disable  in production