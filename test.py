from flask import Flask
from cnn import search as CNNSearch

app = Flask(__name__)

app.config['SECRET_KEY'] = "secret_key"

@app.route("/cnn")
def cnn():
    print(CNNSearch(r"C:\Users\PND280\Documents\GitHub\CS336_M11.KHCL\query_img\all_souls_1.jpg"))
    return "OK"

if __name__ == "__main__":
    app.run(debug=True)