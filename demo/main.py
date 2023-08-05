import cv2
import tensorflow as tf
from vision_function import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#set up 
mlpc = get_model_mlpc() 
mrc = get_model_mrc()
mrp = get_model_mpc()
mrp_IN = get_model_mpc_IN()
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW) #define capture obj

while (True):
    ret,frame = vid.read()
    result,frame = read_lisense_plate(frame,mrc,mrp_IN,mlpc,show_result= True)
    print(result)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv2.destroyAllWindows()