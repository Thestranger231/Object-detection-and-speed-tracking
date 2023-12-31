import cv2
import math
import json
import numpy as np
import cvzone
from sort import *
from ultralytics import YOLO

with open("video.json") as video_file:  # it open the video.json file where video name is present 
    video = json.load(video_file) #it load data from json file which contain the file name 
    video_path = video["video_path"] # video_path save the name of video file


cap = cv2.VideoCapture(video_path) # for recorded video
cap.set(3, 480)  # set the frame width of video
cap.set(4, 240)  # set the frame height of video

model = YOLO("yolov8n.pt")  # yolo version and technology

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
              "dining-table", "toilet", "monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]  # classname database is coco dataset

output_file = "output.txt"  # Name of the text file to save the output

coordinates = {'x1': None, 'y1': None, 'x2': None, 'y2': None}  # this line creates the file with 4 keys
file_path = 'coordinates.json'  # file name
frame = None
callback_set = False  # call back function


def save_coordinates_to_json(file_path, coordinates):  # this saves coordinates to the file
    with open(file_path, 'w') as json_file:  # this file opens the file with write mode
        json.dump(coordinates, json_file) # it save the data to coordinates.json file 


def mouse_callback(event, x, y, flags, param):  # mouse callback function
    global coordinates, frame
    if event == cv2.EVENT_LBUTTONDOWN:  #it use for mouse function
        coordinates['x1'], coordinates['y1'] = x, y #  save the coordinates 
    elif event == cv2.EVENT_LBUTTONUP: 
        coordinates['x2'], coordinates['y2'] = x, y
        cv2.rectangle(frame, (coordinates['x1'], coordinates['y1']), (coordinates['x2'], coordinates['y2']),
                      (0, 255, 0), 2)
        cv2.imshow('Video Stream', frame)  # it streams the file to select coordinates by the user
        save_coordinates_to_json(file_path, coordinates)  # save coordinates to the file
        print(f"Coordinates saved to '{file_path}' successfully!")  # it prints after saving coordinates to the file


def main():
    global frame, callback_set
    cap = cv2.VideoCapture(video_path)  # where we select which video is used to play and select coordinates

    while True:
        ret, frame = cap.read()  # read the next frame
        if not ret:  # if the frame is not captured
            break

        cv2.imshow('Video Stream', frame)  # display the current frame

        # Check if the callback is set, if not, set the callback
        if not callback_set:
            cv2.setMouseCallback('Video Stream', mouse_callback)
            callback_set = True

        key = cv2.waitKey(0)
        if key == 27:  # Press 'Esc' to exit the program 
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

with open("coordinates.json") as coord_file:  # it opens the coordinate file
    try:
        coordinates_data = json.load(coord_file)  # it loads data of the coordinates

        # Check if the coordinates_data is a dictionary with keys 'x1', 'y1', 'x2', 'y2'
        if isinstance(coordinates_data, dict) and all(key in coordinates_data for key in ["x1", "y1", "x2", "y2"]):
            # Extract coordinates from the dictionary
            coordinates = [[coordinates_data["x1"], coordinates_data["y1"], coordinates_data["x2"], coordinates_data["y2"]]]
        else:
            raise ValueError("Invalid format in coordinates.json. Expecting a dictionary with keys 'x1', 'y1', 'x2', 'y2'.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing coordinates.json: {e}")  # if coordinates are not proper, then it displays an error message


with open("conf.json") as conf_file:  # it opens the file in read mode
    conf = json.load(conf_file)  # it loads data from the file
    confidence_threshold = conf["confidence_threshold"]  # it sets the value of confidence_threshold from the file

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)  # it is used for tracking

with open("coordinates.json") as coord_file:  # it opens the coordinate file
    try:
        limit_data = json.load(coord_file)  # it loads data of the limit

        # Check if the limit_data is a dictionary with keys 'x1', 'y1', 'x2', 'y2'
        if isinstance(limit_data, dict) and all(key in limit_data for key in ["x1", "y1", "x2", "y2"]):
            # Extract limit from the dictionary
            limit = [[limit_data["x1"], limit_data["y1"], limit_data["x2"], limit_data["y2"]]]
            x1_limit = int(limit_data["x1"]) # limit[0]
            y1_limit = int(limit_data["y1"]) # limit[1]
            x2_limit = int(limit_data["x2"]) # limit[2]
            y2_limit = int(limit_data["y2"]) # limit[3]
        
        else:
            raise ValueError("Invalid format in limit.json. Expecting a dictionary with keys 'x1', 'y1', 'x2', 'y2'.") # if user set the invalid rectangle then it error message will display 
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing limit.json: {e}")  # if limit are not proper, then it displays an error message

totalcount = [] # we use this for counting of total count 
carcount = [] # we use this for counting only car how many cars we detect in the selected region 
bikecount = [] # we use this for counting only bike how many cars we detect in the selected region 
buscount = [] # we use this for counting only buses how many cars we detect in the selected region 
prev_positions = {} # we use this for detecting the speed it is intinal position of vehicle while detecting 
prev_timestamps = {} # we use this for detecting the speed it is initinal time of vehicle while detecting 
scale_factor = 60 / (1000 * 1000) # it is scaler factor use for mesuring the distance that was in pixel convert pixels to metre

frame_rate = 60 # set desire frame rate

while True:
    success, img = cap.read()  # it captures the image

    
    # imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED) # this both used for graphics the image over count
    # img = cvzone.overlayPNG(img, imgGraphics, (0,0))
        
    if not success or img is None:
     break
    with open(output_file, 'a') as f:  # Open the file in each iteration to overwrite previous results
        for coord in coordinates: #loop for coordinates
            x1, y1, x2, y2 = coord # x1, y1, x2, y2 are coordinate set by user
            w, h = x2 - x1, y2 - y1 # it set the height and width
            imgRegion = img[y1:y2, x1:x2]# Crop the region of interest from the image using the given coordinates
            results = model(imgRegion, stream=True)  # it sets the region
            detections = np.empty((0, 5))  # it empties the detection list
            for r in results: #loop for results
                boxes = r.boxes
                # bounding box
                for box in boxes:
                    # Bounding Box
                    bx1, by1, bx2, by2 = box.xyxy[0]
                    bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)
                    bw, bh = bx2 - bx1, by2 - by1

                    # confidence level
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # class name
                    cls = int(box.cls[0])
                    if conf >= confidence_threshold:  # it reads data from conf.json and sets an if statement
                        cv2.rectangle(imgRegion, (bx1, by1), (bx2, by2), (0, 255, 0), 2)  # it sets the rectangle in the frame
                        cv2.putText(imgRegion, f'{classNames[cls]} {conf}', (max(0, bx1), max(35, by1)),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)  # it displays some text above the box
                        currentArray = np.array([bx1, by1, bx2, by2, conf])  # array of detection
                        detections = np.vstack((detections, currentArray))  # it updates the detection list
                    else:
                        print("")

                    # Write bounding box information to the text file
                    f.write(f"{classNames[cls]} {conf} {bx1} {by1} {bx2} {by2}\n")

            # Paste the processed region back to the original image
            img[y1:y2, x1:x2] = imgRegion

            resultsTracker = tracker.update(detections)  # it sets resultsTracker from the detection list
           # cv2.line(img, (x1_limit, y1_limit), (x2_limit, y1_limit), (0,0,255), 2)
            for result in resultsTracker:  # it starts the loop in results
                bx1, by1, bx2, by2, ID = result  # it sets results in 5 keys
                bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2) # it convert the bx1, by1, bx2, by2 convert into intger
                # Calculate the width and height for bounding box
                bw, bh = bx2 - bx1, by2 - by1
                # Draw the corner rectangle and put text
                cv2.rectangle(imgRegion, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
                text_position_x = max(0, bx1) # it set the text position of x
                text_position_y = max(by1 - 10, 0) # it set the test poistion of y
                cv2.putText(imgRegion, f'{int(ID)}', (text_position_x, text_position_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) # it put text over the box
                cx,cy = bx1+bw//2, by1+bh//2 #it set cx and cy for inside the deteect box
                dx, dy = x1_limit+bw//2, y1_limit+bh//2 # it is use for counting it create the virtual line when any vehicle cross the line it update the count 
                cv2.circle(imgRegion, (cx, cy), 5, (2500,0,255), cv2.FILLED) # it create a small dot over the box
                class_name = classNames[cls] # it set the class names
                conf = round(conf, 2)  # it round off the confdinece value 
                if ID in prev_positions and ID in prev_timestamps: # it check the ID in position and timestamps
                 prev_x, prev_y = prev_positions[ID] # it set the prev_x and prev_y
                 prev_timestamp = prev_timestamps[ID] # it set timestamaps for every id
                 # Calculate distance traveled in pixels
                 distance = math.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2)
                 # Get the time elapsed in seconds
                 current_time = time.time()
                 elapsed_time = current_time - prev_timestamp
                 # Calculate speed in pixels per second
                 speed = distance / elapsed_time # it calclute the speed 
                 speed_kph = speed * scale_factor  # it convert the speed 
                 cv2.putText(imgRegion, f'Speed: {speed:.2f} km/h', (text_position_x, text_position_y + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA) # it show the speed
                 # Update the previous timestamp and poistions
                 prev_positions[ID] = (cx, cy)
                 prev_timestamps[ID] = current_time
                else:
                    prev_positions[ID] = (dx, dy)
                    prev_timestamps[ID] = time.time()
                    print("")
                if  x1_limit <= dx <= x2_limit and y1_limit <= dy <= y2_limit: # it check the vehicle cross the virtual line or not
                 if totalcount.count(ID) ==0: # it check detect the count ID aldready detect or not
                  totalcount.append(ID) # if ID is not count it update the count
                  if  classNames[cls] == "car" and carcount.count(ID) == 0: # check the detect vehicle is car or not and detect vehicle is aldready count or not
                       carcount.append(ID) # if ID is count then it update the count 
                  elif classNames[cls] == "bike" or classNames[cls] == "motorbike" and bikecount.count(ID) == 0:# check the detect vehicle is bike or not and detect vehicle is aldready count or not
                        bikecount.append(ID) # if ID is count then it update the count 
                  elif  classNames[cls] == "bus" and buscount.count(ID) == 0:# check the detect vehicle is bus or not and detect vehicle is aldready count or not
                     buscount.append(ID)   # if ID is count then it update the count                  
                cv2.putText(img, f"Cars: {len(carcount)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # it display the car count and keep updating 
                cv2.putText(img, f"Bikes: {len(bikecount)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # it display the bike count and keep updating 
                cv2.putText(img, f"Buses: {len(buscount)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2) # it display the bus count and keep updating

           
            else:
                print("")
    
    cv2.imshow("Video Stream", img)  # it displays the frame with tracking info
    if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):  # Press 'q' to exit
        break

# Release the capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()