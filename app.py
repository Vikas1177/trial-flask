from flask import Flask , render_template , request

import pickle
tokenizer = pickle.load(open("models/cv.pkl", "rb"))
model = pickle.load(open("models/clf.pkl", "rb"))

app = Flask(__name__)



# @app.route('/' , methods=["GET","POST"])
# def hello_world():
    # return 'Hello, World!'
    # text = ""                     #this was used before making predict route just post and get the result displayed in webpage
    # if request.method == 'POST':
    #     text = request.form.get('email-content')
    # return render_template('index.html', text=text)

@app.route('/')
def home():
    return render_template('index.html')           

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
      email_text= request.form.get('email-content')
    tokenized_email = tokenizer.transform([email_text])
    predictions = model.predict(tokenized_email)
    predictions = 1 if predictions == 1 else -1
    return render_template('index.html', predictions=predictions , text=email_text)

if __name__ =="__main__":
    app.run(debug=True)


