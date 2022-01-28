import os
import cv2
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from demo import retrieve_img_resnet
import numpy as np 

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowedFile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('/index.html')


@app.route('/', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowedFile(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display(filename):
    #print('display filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/crop', methods=['POST'])
def crop():
    data = dict(request.form)
    filePath = url_for('static', filename='uploads/' + data["file"])
    filePathNorm = filePath.replace("/", "\\\\")[2:]

    if (data["x"] != "-1"):
        img = cv2.imread(filePath.replace("/", "\\\\")[2:])

        startX = int(data["x"])
        endX = int(data["x"]) + int(data["w"])
        startY = int(data["y"])
        endY = int(data["y"]) + int(data["h"])

        crop_img = img[startY:endY, startX:endX]
        cv2.imwrite(filePathNorm, crop_img)

    return ""


@app.route('/search/<filename>')
def search(filename):
    print(filename)
    feature_path="feature/RESNET.npz"
    img_path="data/oxford5k_images/"+filename
    input_path="data/oxford5k_images/"
    data = np.load(feature_path,  allow_pickle=True)
    features_storage = data['features']
    list_result=retrieve_img_resnet(img_path,features_storage,input_path)
    list_result=[i.split("\\")[-1] for i in list_result]
    return render_template('search.html', filename=filename,scores=list_result)

if __name__ == "__main__":
    app.run(debug=True)
