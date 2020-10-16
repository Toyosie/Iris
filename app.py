from flask import Flask,render_template,request,url_for
import pickle
import numpy as np

model = pickle.load(open('iris.pkl','rb'))

app = Flask(
    __name__,
    template_folder='template'
)

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['sl']
    data2 = request.form['sw']
    data3 = request.form['pl']
    data4 = request.form['pw']
    arr = np.array([[data1,data2,data3,data4]])
    prediction = model.predict(arr)
    return render_template('predict.html', data=prediction)


if __name__ == "__main__":
    app.run(debug=True)