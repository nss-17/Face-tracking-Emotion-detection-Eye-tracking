from flask import Blueprint, render_template, request, flash, jsonify, send_from_directory, redirect, url_for, Response, current_app
import requests
from flask_login import login_required, current_user
from .models import Patient
from . import db
import json
import cv2
import datetime, time
import os, sys
from threading import Thread
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array


views = Blueprint('views', __name__)

camera = cv2.VideoCapture(0)

global rec_frame, rec, out 
rec=0
global temp_patient_id
temp_patient_id = 0
global temp_path # store video_path after recorded
temp_path = ''

# Variable for moving sence, use in func task()
global temp
temp = 0

# Choose eye or face feature 
global option
option = ''

global result_text_temp
result_text_temp = ''


def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)
            
                
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass

def update_diagnostic_result(app, result):
    with app.app_context():
        try:
            patient = Patient.query.get(temp_patient_id)
            if patient:
                patient.diagnostic = result
                db.session.commit()
                print(f'Diagnostic result updated to: {result}')
            else:
                print('Patient not found')
        except Exception as e:
            print(f'Error updating diagnostic result: {str(e)}')

@views.route('/')
def home():
    return render_template("home.html")

# Information form
@views.route('/patient')
def input_info():
    return render_template('input_info.html', user=current_user)
# Process after input info
@views.route('/process_input', methods=['POST'])
def process_input():
    _name = request.form.get('name')
    _age = request.form.get('age')
    _gender = request.form.get('gender')
    _diagnostic = 'temp' # update later
    
    # choose AI model (1 = face, 2 = eye)
    option = request.form.get('model')
    print('option from input form')
    print(option)
    # Store input infor to database
    new_patient = Patient(name = _name, age = _age, gender = _gender, diagnostic = _diagnostic, user_id=current_user.id)
    db.session.add(new_patient) # adding new patient to the database 
    db.session.commit()
    flash('New patient added!', category='success')
    print('patient id: ', new_patient.id)
    global temp_patient_id
    temp_patient_id = new_patient.id

    return redirect(url_for('views.index', option = option))  

@views.route('/main_page/<option>')
def index(option):
    return render_template('index.html', option=option)

@views.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@views.route('/preload_video/<option>/<filename>')
def preload_video(option, filename):
    print('option from preload_video func')
    print(option)
    if option == 'model1':
        print('face')
        return send_from_directory('./preload_video/model1', filename)
    elif option == 'model2':
        print('eye')
        return send_from_directory('./preload_video/model2', filename)
    else:
        # Handle the case when option is neither 'model1' nor 'model2'
        return "Invalid option", 404


@views.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':   
        if  request.form.get('rec') == 'Start/Stop Recording':
            # global temp
            # temp += 1 # if press the btn first time
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_path = os.path.join('website/videos', f'{temp_patient_id}.avi')
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()

                # store the video path to session
                print(video_path)
                global temp_path
                temp_path = video_path
                
            elif(rec==False):
                out.release()
            
            return jsonify({'status': 'success'})
              
    elif request.method=='GET':
        return render_template('index.html', option=option)
    # Move to result page if press "start/stop" button again
    # if(temp ==2):
    #     temp = 0
    #     return redirect(url_for('views.result_page'))
    
    return jsonify({'status': 'invalid request'})
    #return render_template('index.html', option=option)

# result page
@views.route('/result_page')
def result_page():
    return render_template('result.html')

def gen():
    model = load_model('Emotion_Detection.h5')

    # Retrieve user information and video path from the session
    # user_info = session.get('user_info', {})
    # print(user_info.get('name','no name'))
    # video_path = session.get('recorded_video_path', '')

    # if not video_path:
    #     # Handle the case when the video path is not available
    #     return
    
    video_path = temp_path
    # video_path = './0.avi'
    video = cv2.VideoCapture(video_path)
    # video = cv2.VideoCapture(0)
    face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    class_labels = ['tuc gian','vui ve','binh thuong','buon','bat ngo']

    status = []

    
    while True:
        # Lấy một khung hình video
        ret, frame = video.read()
        if ret is False or frame is None:  # Kiểm tra xem frame có giá trị hay không
            break
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

    # ve hinh chu nhat xung quanh khuon mat
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

    # neu vung trong tam co gia tri thi convert sang mang
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

    # nhan dien cam xuc
                preds = model.predict(roi)[0]
                label=class_labels[preds.argmax()]
                status.append(preds.argmax())
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            print("\n\n")

        # Delay bc the frame is running fast on web page
        time.sleep(0.05)  

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cnt_status = len(set(status))
    global result_text_temp
    if cnt_status >=3:
        result_text_temp = "Signs of autism"
        print("Have signs of autism !!")
        # Update the database with the diagnostic result
        #update_diagnostic_result('Signs of autism')
    else:
        result_text_temp = "Normal"
        print("Normal")
        # Update the database with the diagnostic result
        #update_diagnostic_result('Normal')

# def update_diagnostic_result(diagnostic_result):
#     with current_app.app_context():
#         # Update the diagnostic result for the patient in the database
#         patient = Patient.query.get(temp_patient_id)
#         if patient:
#             patient.diagnostic = diagnostic_result
#             db.session.commit()

@views.route('/result_video')
def result_video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@views.route('/result_text')
def result_text():
    # Update the diagnostic result in the database when the button is clicked
    update_diagnostic_result(current_app._get_current_object(), result_text_temp)
    
    # Retrieve the result text 
    _result_text = result_text_temp
    return render_template('result_text.html', result_text= _result_text)

