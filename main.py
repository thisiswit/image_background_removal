from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import inference
import os
from PIL import Image
from inference import x
import io
import requests

application = Flask(__name__)
app=application
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html", error_image=False)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        urlImage = request.form['urlImage']
        fileBackground = request.files['fileBackground']
        urlBackground = request.form['urlImageBackground']

        if fileBackground or urlBackground and file or urlImage:
            if fileBackground or urlBackground:
                if fileBackground:
                    imagem = Image.open(io.BytesIO(fileBackground.read()))
                
                else:
                    response = requests.get(urlBackground)
                    imagem = Image.open(io.BytesIO(response.content))
                
                backgroundPath = os.path.join(app.config['UPLOAD_FOLDER'], 'background.png')
                imagem.save(backgroundPath, format='PNG')
            
            if file or urlImage:
                if file:
                    imagem = Image.open(io.BytesIO(file.read()))
                
                else:
                    response = requests.get(urlImage)
                    imagem = Image.open(io.BytesIO(response.content))
                
                imagePath = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
                imagem.save(imagePath, format='PNG')
                inference.predict(imagePath)

                return render_template("uploaded.html", date_time=x)
            
        if fileBackground or urlBackground:
            error= 'Nenhuma imagem para remoção do fundo encontrada, tente novamente!'
        elif file or urlImage:
            error= 'Nenhuma imagem de background encontrada, tente novamente!'
        else:
            error= 'Nenhuma imagem encontrada, tente novamente!'

        return render_template("index.html", error_image=True, error=error)

# Removendo o cache dos endpoints.
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(debug=True)
