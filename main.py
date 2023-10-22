import json
import os
from bson import ObjectId
from flask import Flask, flash, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
from BLL import usersBLL
from colorize import pix2pix
from PIL import Image
from torchvision.utils import save_image

upload_folder: str = r'C:\Users\ASUS\PycharmProjects\ImageColorization\uploads'
# ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
usersBLL = usersBLL.UsersBLL()

app = Flask(__name__)
CORS(app)


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        return super(MyEncoder, self).default(obj)


app.json_encoder = MyEncoder


@app.route('/', methods=['GET'])
def upload():
    return "hye"


@app.route('/users', methods=['GET'])
def get_all_customers():
    users = usersBLL.get_all_users()
    return jsonify(users)


@app.route('/user/<string:id>', methods=['GET'])
def get_by_id(id):
    users = usersBLL.get_user_by_id(id)
    return jsonify(users)


@app.route('/user/<string:name>/<string:password>', methods=['GET'])
def get_user_by_mail_and_password(name, password):
    user = usersBLL.get_user_by_id_and_password(name, password)
    return jsonify(user)


@app.route('/addUser', methods=['POST'])
def add_user():
    usersBLL.addUser(request)
    return jsonify("worked!")

@app.route('/updateUser', methods=['PUT'])
def updateUser():
    usersBLL.updateUser(request)
    return jsonify("updated")


@app.route('/deleteImage/<string:id>/<string:path>', methods=['DELETE'])
def deleteImage(id, path):
    u = get_by_id(id)
    file = os.path.join(r"C:\Users\ASUS\imageColorization\src\assets", u.json['folderPath'],path)
    print(file)
    os.remove(file)
    return jsonify("deleted!")


@app.route('/user/getImagesUrl/<string:id>', methods=['GET'])
def getImagesUrl(id):
    return jsonify(os.listdir(os.path.join(r"C:\Users\ASUS\imageColorization\src\assets\users", id)))


@app.route('/user/getGalleryUrl', methods=['GET'])
def getGalleryUrl():
    return jsonify(os.listdir(r"C:\Users\ASUS\imageColorization\src\assets\gallery"))


@app.route('/colorize/<string:id>', methods=['POST'])
def upload_file(id):
    user = get_by_id(id)
    print(request.files)
    # check if the post request has the file part
    if 'file' not in request.files:
        print('no file in request')
        return ""
    file = request.files['file']
    if file.filename == '':
        print('no selected file')
        return ""
    if file:  # and allowed_file(file.filename):
        print("hello")
        filename = secure_filename(file.filename)
        # file.save(os.path.join(upload_folder, filename))
        print(user.json['folderPath'])
        file.save(r"C:\Users\ASUS\PycharmProjects\ImageColorization\results\temp.png")
        image = Image.open(r"C:\Users\ASUS\PycharmProjects\ImageColorization\results\temp.png")
        image = pix2pix(image)
        save_image(image, os.path.join(r"C:\Users\ASUS\imageColorization\src\assets", user.json['folderPath'],
                                       secure_filename(file.filename)))
        return jsonify(os.path.join(user.json['folderPath'], secure_filename(file.filename)))
    print("end")
    return ""


app.run(debug=True)
