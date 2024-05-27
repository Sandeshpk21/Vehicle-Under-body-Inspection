from ultralytics import YOLO
from PIL import Image
import cv2
import os
import time
from datetime import datetime
import json

def load_roi_definitions_from_json(json_file):
    with open(json_file) as f:
        roi_definitions = json.load(f)
    return roi_definitions

# camera_frames_dict = {
#      "Cam-1": cv2.imread(r"C:\Underbody\Dataset\Cam-1\cam1_01_02_24_14_27_40.jpg"),
#      "Cam-2": cv2.imread(r"C:\Underbody\Dataset\Cam-2\cam2_01_02_24_14_27_40.jpg"),
#      "Cam-3": cv2.imread(r"C:\Underbody\Dataset\Cam-3\cam3_01_02_24_14_27_40.jpg"),
#      "Cam-4": cv2.imread(r"C:\Underbody\Dataset\Cam-4\cam4_01_02_24_14_27_40.jpg"),
#      "Cam-5": cv2.imread(r"C:\Underbody\Dataset\Cam-5\cam5_01_02_24_14_27_40.jpg"),
#      "Cam-6": cv2.imread(r"C:\Underbody\Dataset\Cam-2\cam2_01_02_24_14_27_40.jpg")
#      }

def assign_part_labels_from_yolo_with_camera(camera_frames_dict):
    os.chdir(r'C:\Underbody')
    model = YOLO('best_epoch1046.pt')
    roi_definitions_cam3 = load_roi_definitions_from_json(r'C:\Underbody\roi_definitions_cam3.json')
    #print("ROI:", roi_definitions_cam3)
    roi_definitions_cam4 = load_roi_definitions_from_json(r'C:\Underbody\roi_definitions_cam4.json')
    roi_definitions_cam1 = load_roi_definitions_from_json(r'C:\Underbody\roi_definitions_cam1.json')
    roi_definitions_cam2 = load_roi_definitions_from_json(r'C:\Underbody\roi_definitions_cam2.json')
    roi_definitions_cam5 = load_roi_definitions_from_json(r'C:\Underbody\roi_definitions_cam5.json')
    class_to_part_mapping_cam5 = load_roi_definitions_from_json(r'C:\Underbody\class_to_part_mapping_cam5.json')
    assigned_labels_per_frame = {}
    i = 0
    temp = 'run_' + str(int(time.time()))
    run_folder = temp
    os.mkdir(run_folder)
    os.chdir(run_folder)
    #print("Camera\n" , camera_frames_dict)
    for camera_key, camera_frames in camera_frames_dict.items():
        #print(f"Processing camera {camera_key} \n , camera_frames {camera_frames}")
        assigned_labels = {}
        results_list = [model(camera_frames)]
        
        
        
        if isinstance(results_list, list):
           
           for result_list in results_list:
             for result in result_list:
                 j = 0
                 all_boxes = result.boxes.data
                 boxes_1 = result.boxes
                 class_ids = boxes_1.cls
                 class_names = result.names
                 
                 
                 for i in range(len(all_boxes)):
                     box = all_boxes[i]
                     xmin, ymin, xmax, ymax = box[:4]
                     class_id = class_ids[i].item()
                     class_name = class_names[class_id]
                     if camera_key.startswith("Cam-") and int(camera_key.replace("Cam-", "")) == 1:
                            roi_definitions = roi_definitions_cam1
                            j = 1
                            #print("Camera 1")
                                                       
                     elif  camera_key.startswith("Cam-") and int(camera_key.replace("Cam-", "")) == 2:
                            roi_definitions = roi_definitions_cam2
                            j = 2
                            #print("Camera 2")
                            
                        
                     elif camera_key.startswith("Cam-") and int(camera_key.replace("Cam-", "")) == 5:
                           # print("Outside if")
                            if class_name in class_to_part_mapping_cam5:
                                #print("/n Inside if /n")
                                # print("/n Class name: /n",class_name)
                                #print("/n Class to part mapping: /n",class_to_part_mapping_cam5[class_name])
                                # Use direct class-to-part mapping for unique class names
                                actual_part_name = class_to_part_mapping_cam5[class_name]
                                assigned_labels[actual_part_name] = box
                            else:
                                # Use ROI definitions for common class names
                                roi_definitions = roi_definitions_cam5
                                

                            j = 5
                            #print("Camera 5")
                     else:
                            
                            is_cam_3_frame = camera_key.startswith("Cam-") and int(camera_key.replace("Cam-", "")) == 3
                            is_cam_4_frame = camera_key.startswith("Cam-") and int(camera_key.replace("Cam-", "")) == 4

                            if is_cam_3_frame:
                                roi_definitions = roi_definitions_cam3
                                #frame = result.orig_img
                                j = 3
                                
                            elif is_cam_4_frame:
                                roi_definitions = roi_definitions_cam4
                                #frame = result.orig_img
                                j = 4
                                 
                            else:
                                
                                roi_definitions = {}

                            
                     for roi_label, (actual_part_name, roi_coordinates) in roi_definitions.items():
                            roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_coordinates

                                
                            if (roi_x_min <= xmin <= roi_x_max) and (roi_y_min <= ymin <= roi_y_max):
                                assigned_labels[actual_part_name] = box
                                break  
                        
                        
                 cv2.imwrite(os.path.join(f'original_frame{j}.jpg'), result.orig_img)
                 result.save(filename=os.path.join(f'annotated_frame{j}.jpg'))
                 

                
                 assigned_labels_per_frame[camera_key] = assigned_labels

    return assigned_labels_per_frame, run_folder
    

# assigned_labels, folder = assign_part_labels_from_yolo_with_camera(camera_frames_dict)
# print("Assigned labels", assigned_labels)



  
