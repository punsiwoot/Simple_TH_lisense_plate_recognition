import cv2
import tensorflow as tf
import tensorflow.keras.layers as nn

def process_img(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tensor = tf.constant(img/255)
    tensor = tf.reshape(tensor, (tensor.shape[0],tensor.shape[1],1), name=None)
    tensor = tf.image.resize(tensor,(100,200))
    tensor = tf.expand_dims(tensor,axis=0)
    return tensor

def get_model_mlpc():
    model = tf.keras.Sequential([
    # nn.Rescaling(scale=1./255, offset=0.0),
    nn.Conv2D(filters=32, kernel_size = (10,10), activation='relu', input_shape=(100, 200, 1)),
    nn.Conv2D(filters=16, kernel_size = (5,5), activation='relu'),
    nn.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None),
    nn.Flatten(),
    nn.Dense(30,activation='relu'),
    nn.Dense(10,activation="relu"),
    nn.Dense(1)
    ])
    model.load_weights("weight/clp/weight")
    input_signature = [tf.TensorSpec(shape=(1,100,200,1), dtype=tf.float32)]
    model_fn = tf.function(input_signature=input_signature)(model.call)
    return  model_fn


# Make predictions using the optimized function
# predictions = predict_fn(input_data)
# input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32)]
# predict_fn = tf.function(input_signature=input_signature)(model.call)

def local_plate(image_name, input_form="PATH", DCN_filter = False , Model = None, detect_thresho = 0.6):
    keep_box_coor = []
    keep_crop_image = []
    if input_form == "PATH":
        image = cv2.imread(str(image_name)) # hard [4 40 44 38(stuck) 27 23]
    elif input_form == "IMG":
        image = image_name
    image = cv2.resize(image,[500,500])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
    canny = cv2.Canny(blur, 60, 100)
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Iterate thorugh contours and draw rectangles around contours
    # print(f"num of box : {len(cnts)}")
    count = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if (w>h) and (1.5<(w/h)<2.2) and (40<h<190) and (w>140) and (170<(x+(w/2))<330) :  # set filter box coor  (150<(x+(w/2))<350)
            count += 1
            # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            keep_box_coor.append([w,h,x,y])
    keep_box_coor.sort()
    for j in keep_box_coor:
        keep_crop_image.append(gray[j[3]:j[3]+j[1],j[2]:j[2]+j[0]])

    if DCN_filter and Model is not None :
        # print("in")
        for i in range(len(keep_crop_image)):
            result =  Model(process_img(keep_crop_image[i]))[0][0]
            # try : result =  Model.predict(process_img(keep_crop_image[i]),verbose=0)[0][0]
            # except : print("Model predict eror")
            # print(result)
            if result > detect_thresho:
                return [keep_box_coor[i]] ,[keep_crop_image[i]], image
        return [],[],image



    # print(f"number of box that found : {count}")
    # cv2.imshow('canny', canny)
    # cv2.imshow('image', image)
    # cv2.imshow("blur",blur)
    # cv2.waitKey(0)
    return keep_box_coor ,keep_crop_image, image

def local_CP(img,input_form = "PATH"):
    if input_form == "PATH":
        image = cv2.imread(str(image_name)) # hard [4 40 44 38(stuck) 27 23]
    elif input_form == "IMG":
        image = img
    
