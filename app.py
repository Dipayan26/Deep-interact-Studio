from flask import Flask
from flask import Flask, render_template, request, url_for
# from flask import Flask, request, render_template, redirect, url_for, flash

import os

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template("index.html")



if __name__ == '__main__':
    app.run(port=5008, debug=True)