from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, Response
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_sockets import Sockets
from ultralytics import YOLO
from gevent import pywsgi, monkey
from geventwebsocket.handler import WebSocketHandler
from mapping import assign_part_labels_from_yolo_with_camera
import cv2
import base64
import json
import threading
import time
import numpy as np
import torch
import os
import time
from datetime import datetime
import pandas as pd

monkey.patch_all()

app = Flask(__name__)
sockets = Sockets(app)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:2725@localhost/postgres'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '12345'  # Change this to a strong, random value
db = SQLAlchemy(app)

class users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(40), unique=True, nullable=False)
    password = db.Column(db.String(20), nullable=False)
 
class parts(db.Model):
    part_id = db.Column(db.Integer, primary_key=True)
    part_quant = db.Column(db.Integer, nullable=False)
    part_name = db.Column(db.String(50), unique=True, nullable=False)
    part_cam = db.Column(db.String(50), nullable=False)

def run_server():
    # Define a global flag variable
    global flag
    flag = False
    print("Server started")
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app, handler_class=WebSocketHandler)
    print("Server executed")
    server.serve_forever()

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    try:
        print("login called")
        username = request.json.get('username')
        password = request.json.get('password')

        print(username)
        print(password)
        
        # Assuming `users` is your SQLAlchemy model
        user = users.query.filter_by(username=username).first()
        print(user)

        if user and user.password == password:
            # session['user_id'] = user.id
            # flash('Login successful!', 'success')
            return jsonify({'message': 'Login successful'}), 200
        else:
            # flash('Invalid credentials. Please try again.', 'error')
            return jsonify({'error': 'Invalid credentials'}), 4011

    except Exception as e:
        # Handle any exceptions that might occur during the login process
        print("An error occurred during login:", str(e))
        return jsonify({'error': 'An error occurred during login'}), 500, False

class DataFetcher:
    def fetch_data(self):
        try:
            data = parts.query.all()
            result_dict = {}
            for row in data:
                part_cam = row.part_cam
                if part_cam not in result_dict:
                    result_dict[part_cam] = []
                result_dict[part_cam].append({
                    'part_id': row.part_id,
                    'part_name': row.part_name,
                    'part_quant': row.part_quant,
                    'part_cam': row.part_cam
                })
            return result_dict
        except Exception as e:
            response = {'error': str(e)}
            return response

@app.route('/fetch_data', methods=["GET"])
def fetch_db():
    # if 'user_id' in session:
            try:
                return jsonify(data)

            except Exception as e:
                response = {'error': str(e)}
                with app.app_context():
                    return jsonify(response), 500
    # else:
    #     flash('Please login once again','error')
    #     return jsonify({'error':'login session timeout'})

class CameraFrameHandler:
    def __init__(self, cam_list):
        self.video_paths = cam_list
        # self.video_paths = video_paths
        self.frames = {}
        # self.model = YOLO(r"C:\Underbody\best_epoch1046.pt")

    def check_connection(self):
        for key, path in self.video_paths.items():
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Error: Could not open camera at {path}.")
                return False
        return True
    
    def get_camera_frames(self):
        for key, path in self.video_paths.items():
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Error: Could not open camera at {path}.")
                return False
            ret, frame = cap.read()
            cap.release()
            if not ret:
                print(f"Error: Could not capture the first frame from {path}.")
                return False
            self.frames[key] = frame
        return self.frames


    def predict(self):
        # result = self.model(self.frames)
        result = result[0].boxes

        return result

    def get_camera_frames_as_json(self):
        if not self.frames:
            success = self.check_connection()
            if not success:
                return None
            self.frames = self.get_camera_frames()

        base64_images = {}
        for key, frame in self.frames.items():
            # Encode the frame to base64
            retval, buffer = cv2.imencode('.jpg', frame)
            if retval:
                base64_image = base64.b64encode(buffer).decode('utf-8')
                base64_images[key] = base64_image
            else:
                print(f"Error: Could not encode {key} to base64.")

        return base64_images
    
    
    def detect_color(self, image):  # Add self as the first parameter
        # Define the lower and upper bounds for the colors you want to detect (in HSV)
        color_ranges = {
            'orange': ([5, 100, 100], [15, 255, 255]),
            'blue': ([100, 100, 100], [140, 255, 255])
            # Add more colors if needed
        }

        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        #cv2.imshow('Original Image', image)
       # cv2.waitKey(500)
      #  cv2.destroyAllWindows()

        for color_name, (lower_color, upper_color) in color_ranges.items():
            # Create a mask using the defined lower and upper bounds
            mask = cv2.inRange(hsv_image, np.array(lower_color), np.array(upper_color))

            # cv2.imshow("mask", mask)
            # cv2.waitKey(0)

            # Check if any pixel is detected within the specified color range
            if cv2.countNonZero(mask) > 0:
                return color_name, True

        # If no color is detected, return False
        return None, False

    def variant_check(self):  # Add self as the first parameter
        # image = self.frames.get("image1")
        # add bounding box co-ordinates here
        # crop the image using the co-ordinates
        # image = cv2.imread(r"C:\Underbody\Cam-1\cam1_01_02_24_14_27_40.jpg")
        # xmin, ymin, xmax, ymax = 647, 377, 761, 455
        image = cv2.imread(r"C:\Underbody\Dataset\Cam-1\cam1_01_02_24_14_27_40.jpg")
        #cv2.imshow("Original Image", image)
        #cv2.waitKey(500) 
        #cv2.destroyAllWindows()
        xmin, ymin, xmax, ymax = 672, 376, 793, 454
        image = image[ymin:ymax, xmin:xmax]
        detected_color, is_detected = self.detect_color(image)  # Use self to call detect_color
        if is_detected:
            print("Detected color:", detected_color)
            if ((detected_color == 'blue' and variant == 'variant2') or (detected_color == 'orange' and variant == 'variant1')):
                return True
            return detected_color
        else:
            print("No color detected.")
            return None

@app.route('/get_frames')
def get_frames():
    # Get camera frames as JSON
    json_output = frame_handler.get_camera_frames_as_json()
    if json_output is not None:
        return json.dumps(json_output)
    else:
        return jsonify({"error": "Failed to get camera frames."})

global flag
flag =False

@sockets.route('/echo')
def echo_socket(ws):
    print("Socket started")
    while not ws.closed:
        ws.send(json.dumps(flag))
        time.sleep(1)
    
    print("socket stop ...........................")
     
@app.route('/variant', methods = ['POST'])
def variant():
    try:
        print('variant called')
        variant = request.json.get('variant')
       
        print(variant)
       #print(self.results)
        global flag
        print("flag toggled")
        flag = not flag
        
        return jsonify({'message': 'variant recived'}), 200
    except Exception as e:
        print("Error in reciving variant",str(e))
        return jsonify({'error':"Error in reciving variant"}),500,False


class  Results:
    def __init__(self):
        self.results = None
        self.cam_result = None

    def get_results(self):
        self.results = data 
        # camera_frames_dict = {
        # "Cam-1": cv2.imread(r"C:\Underbody\Dataset\Cam-1\cam1_01_02_24_14_27_40.jpg"),
        # "Cam-2": cv2.imread(r"C:\Underbody\Dataset\Cam-2\cam2_01_02_24_14_27_40.jpg"),
        # "Cam-3": cv2.imread(r"C:\Underbody\Dataset\Cam-3\cam3_01_02_24_14_27_40.jpg"),
        # "Cam-4": cv2.imread(r"C:\Underbody\Dataset\Cam-4\cam4_01_02_24_14_27_40.jpg"),
        # "Cam-5": cv2.imread(r"C:\Underbody\Dataset\Cam-5\cam5_01_02_24_14_27_40.jpg"),
        # "Cam-6": cv2.imread(r"C:\Underbody\Dataset\Cam-2\cam2_01_02_24_14_27_40.jpg")
        # }
        temp, self.run_folder = assign_part_labels_from_yolo_with_camera(frame_handler.frames)
        for i in range(len(self.results)):
            for value in self.results[f"Cam-{i+1}"]:
                if  temp.get(f'Cam-{i+1}') is not None:
                    temp1 = temp[f"Cam-{i+1}"]
                    for key in temp1:
                        if value['part_name'] == key:
                            value["present"] = True
                            break
                        else:
                            value["present"] = False
                

                
        # Call the variant_check function
        frame_handler.variant_check()
        #print("Temp", temp) 
        #print("Results:\n ", self.results)
        return jsonify(self.results)
        
    
    def save_results(self):
        temp = self.results  
        folder_name = self.run_folder
        os.chdir(r'C:\Underbody')
        os.chdir(folder_name)
        flat_data = []
        for sublist in temp.values():
            flat_data.extend(sublist)
        # Create a DataFrame
        df = pd.DataFrame(flat_data)

        excel_file_path = "output.xlsx"    
        df.to_excel(excel_file_path, index=False)
        #global flag
        #print("flag toggled")
        #flag = not flag
        return True
    

@app.route('/get_results', methods = ['GET'])
def get_results_routes():
    global flag
    print("flag toggled")
    flag = not flag
    x = obj.get_results()
    return x

@app.route('/save_results', methods = ['GET','POST'])
def save_results_routes():
    #obj1 = Results()
    #obj.results = get_results_routes()
    try:
        print("\n Save results api called")
        result = obj.save_results()
        if result:
            return jsonify({'message': 'Results saved successfully'}), 200
        else:
            return jsonify({'error': 'Failed to save results'}), 500

    except Exception as e:
        # Handle any exceptions that might occur during the process
        return jsonify({'error':'\t Failed to save results'}), 500




class camara_edit:
    def __init__(self):
        self.cam_name = None
        self.cam_username = None
        self.cam_password = None
        self.cam_ip = None
        self.cam_port = None
        pass
    
    def add_cam(self):
        try:
            with open(r'C:\Underbody\rtsp.json', 'r') as file:
                cam_list = json.load(file)

            # cam_list.append(new_entry)
            cam_list[self.cam_name] = f"rtsp://{self.cam_username}:{self.cam_password}@{self.cam_ip}:{self.cam_port}/?h264x=4"
            frame_handler.video_paths = [value for key, value in cam_list.items()]
            print(video_paths)

            with open(r'C:\Underbody\rtsp.json', 'w') as file:
                json.dump(cam_list, file, indent=4)
            file.close()
            return True
        except Exception as e:
            print("Error in adding camara",str(e))
            return jsonify({'error':"Error in adding camera"}),500,False
    
    def edit_cam(self):
        try:
            with open(r'C:\Underbody\rtsp.json', 'r') as file:
                cam_list = json.load(file)
            
            var = self.delete_cam()
            if(var == True):
                cam_list[self.cam_name] = f"rtsp://{self.cam_username}:{self.cam_password}@{self.cam_ip}:{self.cam_port}/?h264x=4"
                frame_handler.video_paths = [value for key, value in cam_list.items()]
                print(video_paths)

                with open(r'C:\Underbody\rtsp.json', 'w') as file:
                    json.dump(cam_list, file, indent=4)
                file.close()
                return True
            else:
                return var
        except Exception as e:
            print("Error in editing camera",str(e))
            return jsonify({'error':"Error in editing camera"}),500,False

    def delete_cam(self):
        try:
            with open(r'C:\Underbody\rtsp.json', 'r') as file:
                cam_list = json.load(file)
            
            del cam_list[self.cam_name]
            frame_handler.video_paths = [value for key, value in cam_list.items()]
            print (video_paths)

            with open(r'C:\Underbody\rtsp.json', 'w') as file:
                json.dump(cam_list, file, indent=4)
            
            file.close()
            return True
        except FileNotFoundError:
            print("File not found. Unable to delete camera.")
            return jsonify({'error':"File not found. Unable to delete camera"}),500,False

        except KeyError:
            print("Camera name not found. Unable to find camera = ",self.cam_name)
            return jsonify({'error':f"Camera name not found. Unable to find camera  =  {self.cam_name}"}),500,False

@app.route('/delete_cam', methods = ['POST'])
def delete_camara():
    try:
        print('delete camera called')
        editcam.cam_name = request.json.get("cam_name")

        try:
            var = editcam.delete_cam()
            if(var == True):
                print("deleted")
                return jsonify({'message': 'cam deleted'}), 200
            return var
        except:
            return var
    except Exception as e:
        print("Error in reciving camara name",str(e))
        return jsonify({'error':"Error in reciving camara name"}),500,False

@app.route('/edit_cam', methods = ['POST','GET'])
def edit_camara():
    try:
        print('edit camara called')
        editcam.cam_name = request.json.get('cam_name')
        editcam.cam_ip = request.json.get('cam_ip')
        editcam.cam_username = request.json.get('cam_username')
        editcam.cam_password = request.json.get('cam_password')
        editcam.cam_port = request.json.get('cam_port')
        try:
            var = editcam.edit_cam()
            if(var == True):
                return jsonify({'message': 'cam edited'}), 200
            return var
        except:
            return var
    except Exception as e:
        print("Error in reciving camara input",str(e))
        return jsonify({'error':"Error in reciving camara input"}),500,False

@app.route ('/add_cam', methods = ['POST'])
def add_camara():
    try:
        print('add camara called')
        editcam.cam_name = request.json.get('cam_name')
        editcam.cam_username = request.json.get('cam_username')
        editcam.cam_password = request.json.get('cam_password') 
        editcam.cam_ip = request.json.get('cam_ip')
        editcam.cam_port = request.json.get('cam_port')
        print(editcam.cam_name, editcam.cam_username, editcam.cam_password, editcam.cam_ip, editcam.cam_port)
        try:
            var = editcam.add_cam()
            if(var == True):
                return jsonify({'message': 'cam added'}), 200
            return var
        except:
            return var        
    except Exception as e:
        print("Error in reciving input",str(e))
        return jsonify({'error':"Error in reciving input"}),500,False

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return jsonify({'message': 'logout successfull'})
  

if __name__ == "__main__":
    print("app run start")
    data_fetcher = DataFetcher()
    data = data_fetcher.fetch_data()
    #print(data)
   
    # create an instance of the class camaraframehandler
    with open(r'C:\Underbody\rtsp.json', 'r') as file:
        cam_list = json.load(file)

    frame_handler = CameraFrameHandler(cam_list)
    video_paths = [value for key, value in cam_list.items()]
    obj = Results()
    file.close()

    # Create an instance of the camara_edit class
    editcam = camara_edit()

    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    # socket_thread = threading.Thread(target=send_flag)
    # socket_thread.start()
    # run_server()


