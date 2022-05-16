import imutils
import cv2
import numpy as np
import pandas
from matplotlib import pyplot as plt
from datetime import datetime
import time as t
import threading
import pypyodbc as odbc

DRIVER = 'SQL Server'
SERVER_NAME = 'ROLLIN\SQLEXPRESS'
DATABASE_NAME = 'motion_detection'

def sketch_transform(image):
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale_blurred = cv2.GaussianBlur(image_grayscale, (7, 7), 0)
    image_canny = cv2.Canny(image_grayscale_blurred, 10, 80)
    _, mask = image_canny_inverted = cv2.threshold(image_canny, 30, 255, cv2.THRESH_BINARY_INV)
    return mask

def new_start():
    s =t.time()
    return s

# =============================================================================
# USER-SET PARAMETERS
# =============================================================================

# Number of frames to pass before changing the frame to compare the current
# frame against
FRAMES_TO_PERSIST = 10

# Minimum boxed area for a detected motion to count as actual motion
# Use to filter out noise or small objects
MIN_SIZE_FOR_MOVEMENT = 2000

# Minimum length of time where no motion is detected it should take
# (in program cycles) for the program to declare that there is no movement
MOVEMENT_DETECTED_PERSISTENCE = 100

# =============================================================================
# CORE PROGRAM
# =============================================================================
def connection_string(driver, server_name, database_name):
    conn_string = f"""
         DRIVER={{{driver}}};
         SERVER={server_name};
         DATABASE={database_name};
       """
    return conn_string


def sql(T, Mov, co, camused):
    conn = odbc.connect(connection_string(DRIVER, SERVER_NAME, DATABASE_NAME))
    cursor = conn.cursor()

    sql1 = "INSERT INTO motion_detection_analytics(Time_Stamp,Movement,Co_ordinates,Camera_used) VALUES('{}','{}','{}','{}')".format(
        T, Mov, co, camused)
    # print(val, 'New data added')
    cursor.execute(sql1)
    conn.commit()
    cursor.close()
    conn.close()


def Main(ip):
    # Create capture object
    cap = cv2.VideoCapture(5)  # Flush the stream
    # cap = cv2.VideoCapture("rtsp://admin:123456@10.239.1.223:554/ch01.264")
    # cap = cv2.VideoCapture("rtsp://admin:123456@10.239.2.213:554/ch01.264")
    cap.release()
    #cap = cv2.VideoCapture(ip)  # Then start the webcam
    cap = cv2.VideoCapture("rtsp://admin:123456@"+ip+":554/ch01.264")
    # cap = cv2.VideoCapture("rtsp://admin:123456@10.239.2.213:554/ch01.264")

    # Init frame variables
    first_frame = None
    next_frame = None

    # Init display font and timeout counters
    font = cv2.FONT_HERSHEY_SIMPLEX
    delay_counter = 0
    movement_persistent_counter = 0

    #########################################
    # Time logs
    # List when any moving object appear
    motion_list = [None, None]

    # Time of movement
    time = []

    # Initializing DataFrame, one column is start
    # time and other column is end time
    df = pandas.DataFrame(columns=["Start", "End"])

    # the output will be written to output.avi
    out = cv2.VideoWriter(
        'output.avi',
        cv2.VideoWriter_fourcc(*'MJPG'),
        15.,
        (640, 480))
    #########################################

    # 386 247 214 203
    # upper_left = (400, 200)
    # bottom_right = (580, 380)
    upper_left = (0, 247)
    bottom_right = (286, 447)

    counter = 0
    start = t.time()
    # LOOP!
    while True:
        motion = 0

        # Set transient motion detected as false
        transient_movement_flag = False

        # Read frame
        ret, frame = cap.read()

        text = "Unoccupied"

        # If there's an error in capturing
        if not ret:
            print("CAPTURE ERROR")
            continue

        # Resize and save a greyscale version of the image
        # frame = imutils.resize(frame, width = 750)
        frame = imutils.resize(frame, width=600)

        ###########################
        # Rectangle marker
        r = cv2.rectangle(frame, upper_left, bottom_right, (100, 50, 200), 5)
        rect_img = frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]]

        sketcher_rect = rect_img
        sketcher_rect = sketch_transform(sketcher_rect)

        # Conversion for 3 channels to put back on original image (streaming)
        sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)

        ###########################

        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #####################################
        gray = cv2.cvtColor(rect_img, cv2.COLOR_BGR2GRAY)

        # Blur it to remove camera noise (reducing false positives)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # If the first frame is nothing, initialise it
        if first_frame is None:
            first_frame = gray
            continue

        delay_counter += 1

        # Otherwise, set the first frame to compare as the previous frame
        # But only if the counter reaches the appriopriate value
        # The delay is to allow relatively slow motions to be counted as large
        # motions if they're spread out far enough
        if delay_counter > FRAMES_TO_PERSIST:
            delay_counter = 0
            first_frame = next_frame

        # Set the next frame to compare (the current frame)
        next_frame = gray

        # Compare the two frames, find the difference
        frame_delta = cv2.absdiff(first_frame, next_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # Fill in holes via dilate(), and find contours of the thesholds
        thresh = cv2.dilate(thresh, None, iterations=2)
        # _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # loop over the contours
        for c in cnts:
            # Save the coordinates of all found contours
            (x, y, w, h) = cv2.boundingRect(c)

            # If the contour is too small, ignore it, otherwise, there's transient
            # movement
            if cv2.contourArea(c) > MIN_SIZE_FOR_MOVEMENT:
                transient_movement_flag = True

                # Draw a rectangle around big enough movements
                ###################################################
                motion = 1
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(rect_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                ###################################################

        ################################################
        # Appending status of motion
        x2 = x + int(w / 2)
        y2 = y + int(h / 2)
        coor = "x: " + str(x2) + ", y: " + str(y2)
        motion_list.append(motion)

        motion_list = motion_list[-2:]

        # Appending Start time of motion
        if motion_list[-1] == 1 and motion_list[-2] == 0:
            time.append(datetime.now())

        # Appending End time of motion
        if motion_list[-1] == 0 and motion_list[-2] == 1:
            time.append(datetime.now())

        ################################################
        # The moment something moves momentarily, reset the persistent
        # movement timer.
        if transient_movement_flag == True:
            movement_persistent_flag = True
            movement_persistent_counter = MOVEMENT_DETECTED_PERSISTENCE

        # As long as there was a recent transient movement, say a movement
        # was detected
        if movement_persistent_counter > 0:
            text = "Movement Detected " + str(movement_persistent_counter)
            movement_persistent_counter -= 1
        else:
            text = "No Movement Detected"
        timestamp = datetime.now().strftime("%d-%m-%y %H:%M")
        global M
        M = movement_persistent_counter
        end = t.time()
        lap = end-start
        if M > 80:
            counter += 1
        else:
            counter = 0

        # print(counter)
        # print('movement:', M, '\ncounter:', counter)
        # if counter > 60 and M > 80:
        #     print("Movement=", M)
        #     cv2.putText(frame, str(text), (10, 35), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        # else:
        #     pass
        '''
        if lap>60:
            if counter>60 and M>80:
                print("Movement=", M)
                cv2.putText(frame, str(text), (10, 35), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                start=new_start()
        '''
        if lap > 10:
            if counter > 60 and M > 80:
                print("Movement=", M, timestamp,"Camera used=",ip)
                cv2.putText(frame, str(text), (10, 35), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                sql(str(timestamp), str(M), str(coor), str(cam))
            else:
                start = new_start()
        else:
            pass

        #print("Motion Detection=",movement_persistent_counter)
        # Print the text on the screen, and display the raw and processed video
        # feeds
        #cv2.putText(frame, str(text), (10, 35), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        # For if you want to show the individual video frames
        #    cv2.imshow("frame", frame)
        #    cv2.imshow("delta", frame_delta)

        # Convert the frame_delta to color for splicing
        frame_delta = cv2.cvtColor(frame_delta, cv2.COLOR_GRAY2BGR)

        frame[upper_left[1]: bottom_right[1], upper_left[0]: bottom_right[0]] = sketcher_rect_rgb

        #############################################
        # Write the output video
        out.write(frame.astype('uint8'))
        #############################################

        # Splice the two video frames together to make one long horizontal one
        # cv2.imshow("frame", np.hstack((frame_delta, frame)))
        #cv2.imshow("frame", frame)

        # Interrupt trigger by pressing q to quit the open CV program
        ch = cv2.waitKey(1)
        if ch & 0xFF == ord('q'):
            if motion == 1:
                time.append(datetime.now())
            break

    # Appending time of motion in DataFrame
    #for i in range(0, len(time), 2):
    #    df = df.append({"Start": time[i], "End": time[i + 1]}, ignore_index=True)

    # Creating a CSV file in which time of movements will be saved
    #df.to_csv("motion_detects.csv")
    #out.release()
    cap.release()

    # Cleanup when closed
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    #Main('hi')
    df = pandas.read_excel('cctv-NVR-3.xlsx', engine='openpyxl')
    for i in range(len(df)):
        ip = str(df['IP Address'][i])
        #Main(str(ip))
        #print(str(ip))
        th2=threading.Thread(target=Main,args=(ip,))
        th2.start()

    while True:
        t.sleep(15)