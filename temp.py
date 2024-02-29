# 30/11

from flask import Flask, render_template, Response, request, send_from_directory, redirect, url_for, session
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import numpy as np
from keras.models import load_model



global rec_frame, rec, out 
rec=0

#make folder directory to save videos
try:
    os.mkdir('./videos')
except OSError as error:
    pass


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')
app.secret_key = 'son'  # Set a secret key for the session


camera = cv2.VideoCapture(0)

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


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/preload_video/<filename>')
def preload_video(filename):
    return send_from_directory('./preload_video', filename)


@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':   
        if  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_path = os.path.join('videos', 'vid_{}.avi'.format(str(now).replace(":",'')))
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


# Information form
@app.route('/input_info')
def input_info():
    return render_template('input_info.html')

# Process after input info
@app.route('/process_input', methods=['POST'])
def process_input():
    name = request.form.get('name')
    age = request.form.get('age')
    gender = request.form.get('gender')

    # Process the input data as needed (like store in a database)

    # Store user information and video path in a session (you may want to use a database)
    session['user_info'] = {'name': name, 'age': age, 'gender': gender}

    return redirect(url_for('result_page'))  # Redirect to the result page after submission

# result page
@app.route('/result_page')
def result_page():
    return render_template('result.html')

def gen():
    model = load_model('model_file_30epochs.h5')

    # Retrieve user information and video path from the session
    # user_info = session.get('user_info', {})
    # video_path = user_info.get('video_path', '')

    # if not video_path:
    #     # Handle the case when the video path is not available
    #     return
    
    video_path = '1.avi'
    video = cv2.VideoCapture(video_path)
    # video = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

    while True:
        ret, frame = video.read()

        # if not ret:
        #     # Break the loop if the video is finished
        #     break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)

        for x, y, w, h in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = normalize.reshape((1, 48, 48, 1))
            result = model.predict(reshaped)
            label = labels_dict[np.argmax(result, axis=1)[0]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/result_video')
def result_video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug = True)
    
camera.release()
cv2.destroyAllWindows()     