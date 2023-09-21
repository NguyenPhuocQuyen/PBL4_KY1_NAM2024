import cv2
import numpy as np 
import ai as ai_module

# Open device at the ID 1 (ID1: Camera computer)
cap = cv2.VideoCapture(0)

# Check whether user-selected camera is opened successfully
if not cap.isOpened():
    print("Could not open video")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    #Check ret can read frame yet?
    if not ret: 
        print("Could not read frame")
        break 
    #resize_frame with dimensions 
    resize_frame = cv2.resize(frame, (1000,700))
    #Display with dimensions resized with title "Display Trash_Classification" 
    cv2.imshow("Display Trash_Classification", resize_frame)
    category = ai_module.classifyObject(frame)
    print(f'Class with the highest probability: {ai_module.Categories(category)}')
    print(f'Object belongs to category number : {category}')

    ## Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()