import cv2
import dlib
import numpy as np
import time
import imutils
from imutils import face_utils
from imutils.video import VideoStream

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

print("WebCam is starting...")

vs = VideoStream(src='crap.avi').start()
time.sleep(2.0)

frame_width = 1024
frame_height = 576


image_points = np.array([
                            (359, 391),     # Nose tip 34
                            (399, 561),     # Chin 9
                            (337, 297),     # Left eye left corner 37
                            (513, 301),     # Right eye right corne 46
                            (345, 465),     # Left Mouth corner 49
                            (453, 469)      # Right mouth corner 55
                        ], dtype="double")

model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip 34
                            (0.0, -330.0, -65.0),        # Chin 9
                            (-225.0, 170.0, -135.0),     # Left eye left corner 37
                            (225.0, 170.0, -135.0),      # Right eye right corne 46
                            (-150.0, -150.0, -125.0),    # Left Mouth corner 49
                            (150.0, -150.0, -125.0)      # Right mouth corner 55

                        ])

while True:
        frame = vs.read()
        # frame = cv2.flip(frame,1)
        frame = imutils.resize(frame, width=1024, height=576)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = gray.shape

        rects = detector(gray, 0)

        if len(rects) > 0:
                text = "{} face(s) found".format(len(rects))
                cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 0, 0), 2)

        for rect in rects:
                        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                        # cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
                
                        shape = predictor(gray, rect)
                        shape = face_utils.shape_to_np(shape)
                
                        for (i, (x, y)) in enumerate(shape):
                                if i == 33:
                                        image_points[0] = np.array([x,y],dtype='double')
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                elif i == 8:
                                        image_points[1] = np.array([x,y],dtype='double')
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                elif i == 36:
                                        image_points[2] = np.array([x,y],dtype='double')
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                elif i == 45:
                                        image_points[3] = np.array([x,y],dtype='double')
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                elif i == 48:
                                        image_points[4] = np.array([x,y],dtype='double')
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                elif i == 54:
                                        image_points[5] = np.array([x,y],dtype='double')
                                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                else:
                                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                        focal_length = size[1]
                        center = (size[1]/2, size[0]/2)
                        camera_matrix = np.array([[focal_length,0,center[0]],[0, focal_length, center[1]],[0,0,1]], dtype="double")

                        dist_coeffs = np.zeros((4,1)) 
                        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)#flags=cv2.CV_ITERATIVE)

                        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                        for p in image_points:
                                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

                        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                        angle = np.arctan2((p2[1]-p1[1]),(p2[0]-p1[0]))
                        angle_degree = round((angle*180)/np.pi,2)
                        # angle = round(np.arctan(slope),2)
                        cv2.putText(frame, str(angle_degree), (bX,bY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        cv2.line(frame, p1, p2, (255,0,0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
                break
#print(image_points)

cv2.destroyAllWindows()
vs.stop()
