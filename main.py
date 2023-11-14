from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import inference
import os
import requests
import urllib.request 
from inference import x

application = Flask(__name__)
app=application
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/")
def index():
    return render_template("index.html", error_image=False)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        urlImage = request.form['urlImage']

        render_template("index.html", error_image=False, loading=True)
        
        if f or urlImage: 
            if f:        
                filename = secure_filename(f.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                f.save(filepath)
                inference.predict(filepath)
                return render_template("uploaded.html", display_detection=filename, fname=filename, date_time=x)
            
            else:
                fname = urlImage.split("/")[-1]
                path = f"static/uploads/{fname}"
                urllib.request.urlretrieve(urlImage, path)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                inference.predict(filepath)
                return render_template("uploaded.html", display_detection=path, fname=fname, date_time=x)
        
        return render_template("index.html", error_image=True)

# Removendo o cache dos endpoints.
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(debug=True)
