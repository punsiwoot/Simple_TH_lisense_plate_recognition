import cv2
import tensorflow as tf
import tensorflow.keras.layers as nn
import matplotlib.pyplot as plt 
from PIL import ImageFont, ImageDraw, Image
import numpy as np

index_char = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23',
            '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
            '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54',
            '6', '7', '8', '9']
to_char = [ '0','1','2','3','4','5','6','7','8','9','ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช',
    'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด',
    'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ',
    'ฟ', 'ภ', 'ม', 'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส',
    'ห', 'ฬ', 'อ', 'ฮ','none']
index_province = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23',
                '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54',
                '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7',
                '70', '71', '72', '73', '74', '75', '76', '77', '8', '9']
index_province_IN = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23',
                '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54',
                '55', '56', '57', '58', '59', '6', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '7',
                '70', '71', '72', '73', '74', '75', '76', '8', '9']
to_province = ['กรุงเทพมหานคร', 'อำนาจเจริญ', 'อ่างทอง', 'บึงกาฬ', 'บุรีรัมย์', 'ฉะเชิงเทรา', 'ชัยนาท', 'ชัยภูมิ', 'จันทบุรี', 'เชียงใหม่', 'เชียงราย', 'ชลบุรี', 'ชุมพร', 'กาฬสินธุ์', 'กำแพงเพชร', 'กาญจนบุรี', 'ขอนแก่น',
                'กระบี่', 'ลำปาง', 'ลำพูน', 'เลย', 'ลพบุรี', 'แม่ฮ่องสอน', 'มหาสารคาม', 'มุกดาหาร', 'นครนายก', 'นครปฐม', 'นครพนม', 'นครราชสีมา', 'นครสวรรค์', 'นครศรีธรรมราช', 'น่าน', 'นราธิวาส', 'หนองบัวลำภู',
                'หนองคาย', 'นนทบุรี', 'ปทุมธานี', 'ปัตตานี', 'พังงา', 'พัทลุง', 'พะเยา', 'เพชรบูรณ์', 'เพชรบุรี', 'พิจิตร', 'พิษณุโลก', 'พระนครศรีอยุธยา', 'แพร่', 'ภูเก็ต', 'ปราจีนบุรี', 'ประจวบคีรีขันธ์', 'ระนอง',
                'ราชบุรี', 'ระยอง', 'ร้อยเอ็ด', 'สระแก้ว', 'สกลนคร', 'สมุทรปราการ', 'สมุทรสาคร', 'สมุทรสงคราม', 'สระบุรี', 'สตูล', 'สิงห์บุรี', 'ศรีสะเกษ', 'สงขลา', 'สุโขทัย', 'สุพรรณบุรี', 'สุราษฎร์ธานี', 'สุรินทร์',
                'ตาก', 'ตรัง', 'ตราด', 'อุบลราชธานี', 'อุดรธานี', 'อุทัยธานี', 'อุตรดิตถ์', 'ยะลา', 'ยโสธร', 'none']

def write_th_front(img,text,x,y):
    fontpath = "front/angsana.ttc"
    font = ImageFont.truetype(fontpath, 45)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y),  str(text), font = font, fill = (0, 255, 0, 0))
    img = np.array(img_pil)
    return img

def process_img(img,Type = "plate",is_gray=False):
    if is_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if Type == "char" :
        ret,img = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
        img = abs(255-img)
        pass
    elif Type == "province":
        ret,img = cv2.threshold(img,110,255,cv2.THRESH_BINARY)
        img = abs(255-img)
        pass


    tensor = tf.constant(img/255)
    tensor = tf.reshape(tensor, (tensor.shape[0],tensor.shape[1],1), name=None)
    if Type == "plate":
        tensor = tf.image.resize(tensor,(100,200))
    elif Type == "char" :
        tensor = tf.image.resize(tensor,(200,100))
    elif Type == "province":
        
        tensor = tf.image.resize(tensor,(100,500))
    tensor = tf.expand_dims(tensor,axis=0)
    return tensor

def get_model_mrc():
    model = tf.keras.Sequential([
    # nn.Rescaling(scale=1./255, offset=0.0),
    nn.Conv2D(filters=40, kernel_size = (5,5), activation='relu', input_shape=(200, 100, 1)),
    nn.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None),
    nn.Conv2D(filters=20, kernel_size = (5,5), activation='relu'),
    nn.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None),
    nn.Flatten(),
    nn.Dense(100,activation='relu'),
    nn.Dense(55,activation='softmax'),
    ])
    model.load_weights("weight/rc/weight")
    input_signature = [tf.TensorSpec(shape=(1,200,100,1), dtype=tf.float32)]
    model_fn = tf.function(input_signature=input_signature)(model.call)
    return  model_fn

def get_model_mpc():
    model = tf.keras.Sequential([
    nn.Conv2D(filters=40, kernel_size = (5,5), activation='relu', input_shape=(100, 500, 1)),
    nn.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None),
    nn.Conv2D(filters=20, kernel_size = (5,5), activation='relu'),
    nn.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None),
    nn.Flatten(),
    nn.Dense(100,activation='relu'),
    nn.Dense(78,activation='softmax'),
    ])
    model.load_weights("weight/rp/weight")
    input_signature = [tf.TensorSpec(shape=(1,100,500,1), dtype=tf.float32)]
    model_fn = tf.function(input_signature=input_signature)(model.call)
    return  model_fn

def get_model_mpc_IN():
    model = tf.keras.Sequential([
    nn.Conv2D(filters=40, kernel_size = (5,5), activation='relu', input_shape=(100, 500, 1)),
    nn.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None),
    nn.Conv2D(filters=20, kernel_size = (5,5), activation='relu'),
    nn.MaxPool2D(pool_size=(2, 2), strides=None, padding="valid", data_format=None),
    nn.Flatten(),
    nn.Dense(100,activation='relu'),
    nn.Dense(77,activation='softmax'),
    ])
    model.load_weights("weight/rp_IN/weight")
    input_signature = [tf.TensorSpec(shape=(1,100,500,1), dtype=tf.float32)]
    model_fn = tf.function(input_signature=input_signature)(model.call)
    return  model_fn

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
    # rule base block method
    # print(f"num of box : {len(cnts)}")
    count = 0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if (w>h) and (1.5<(w/h)<2.2) and (40<h<230) and (w>140) and (170<(x+(w/2))<330) :  # set filter box coor  (150<(x+(w/2))<350)
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
                return [keep_box_coor[i]] ,[keep_crop_image[i]], image , True
        return [],[],image,False



    # print(f"number of box that found : {count}")
    # cv2.imshow('canny', canny)
    # cv2.imshow('image', image)
    # cv2.imshow("blur",blur)
    # cv2.waitKey(0)
    return keep_box_coor ,keep_crop_image, image, False

def local_CP(img,input_form = "PATH"):
    if input_form == "PATH":
        image_lis = cv2.imread(str(img)) # hard [4 40 44 38(stuck) 27 23]
    elif input_form == "IMG":
        image_lis = img

    #tranfrom for method 2
    ret,image_li = cv2.threshold(image_lis,100,255,cv2.THRESH_BINARY) #110 old value
    # ret,image_li_pro = cv2.threshold(image_lis,50,255,cv2.THRESH_BINARY) #110 old value

    # pure_plate_char = image_li[int(image_li.shape[0]*0.10):int(image_li.shape[0]*0.65),int(image_li.shape[1]*0.1):int(image_li.shape[1]*0.9)]
    pure_plate_char = image_li[:int(image_li.shape[0]*0.65),:]
    pure_plate_char_cut = pure_plate_char.copy() #pure_plate[:int(pure_plate.shape[0]*0.65),:]
    threshold_char_h1,threshold_char_h2,threshold_char_w = pure_plate_char.shape[0]*0.5,pure_plate_char.shape[0]*0.9,pure_plate_char.shape[1]*0.2
    # pure_plate_province = image_li[int(image_li.shape[0]*0.65):int(image_li.shape[0]*0.90),int(image_li.shape[1]*0.15):int(image_li.shape[1]*0.85)] #get province
    pure_plate_province = image_lis[int(image_li.shape[0]*0.65):int(image_li.shape[0]*0.91),int(image_li.shape[1]*0.1):int(image_li.shape[1]*0.9)] #get province  

    blurred_pure_plate = pure_plate_char
    # blurred_pure_plate = cv2.GaussianBlur(pure_plate_char, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
    # blurred_pure_plate = cv2.GaussianBlur(pure_plate_char, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)

    canny_blurred_pure_plate = cv2.Canny(blurred_pure_plate, 100, 200)  #50 100
    find_front = cv2.findContours(canny_blurred_pure_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    find_front = find_front[0] if len(find_front) == 2 else find_front[1]
    # count_pp = 0
    keep_coor_raw = []
    keep_coor = []
    for i in find_front:
        x,y,w,h = cv2.boundingRect(i)
        if (h>w)and(threshold_char_h1<h<threshold_char_h2)and((h/w)<5)and((w<threshold_char_w)):
            if ((h/w)>3):
                w = w+int(w*0.2)
                x = x-int(w*0.2/2)
            w=w+int(w*0.15)
            x=x-int(w*0.15/2)
            h=h+int(h*0.05)
            y=y-int(h*0.05/2)
            # count_pp += 1  
            keep_coor_raw.append([x,y,w,h])
            # cv.rectangle(pure_plate_char, (x, y), (x + w, y + h), (36,255,12), 2)
    keep_coor_raw.sort()
    if (len(keep_coor_raw)>=1):
        keep_coor.append(keep_coor_raw[0])
        for i in range(len(keep_coor_raw)-1):
            i = i+1
            if ((keep_coor_raw[i][0]-keep_coor[-1][0]) < keep_coor[-1][2]/2 ):  # if it close togather
                if (keep_coor_raw[i][2]>keep_coor[-1][2]):   # if it more wide
                    keep_coor[-1] = keep_coor_raw[i]
            else : keep_coor.append(keep_coor_raw[i])
    keep_cut_char = []
    for i in range(len(keep_coor)):
        keep_cut_char.append(pure_plate_char_cut[keep_coor[i][1]:keep_coor[i][1]+keep_coor[i][3],keep_coor[i][0]:keep_coor[i][0]+keep_coor[i][2]])
        # cv2.imwrite("crop_CP_test/"+str(i)+".jpg",pure_plate_char_cut[keep_coor[i][1]:keep_coor[i][1]+keep_coor[i][3],keep_coor[i][0]:keep_coor[i][0]+keep_coor[i][2]])
        # cv2.rectangle(pure_plate_char, (keep_coor[i][0], keep_coor[i][1]), (keep_coor[i][0] + keep_coor[i][2], keep_coor[i][1] + keep_coor[i][3]), (36,255,12), 2)
    return keep_cut_char, pure_plate_province , keep_coor

def read_plate(list_char,province,rc,rp,is_gray = False,skip_eror_rc=False,skip_eror_rp=False):
    read_char = ""
    for i in list_char:
        # rc_result = tf.math.argmax(rc(process_img(i,Type='char'))[0])
        if skip_eror_rc:
            try :
                rc_result = to_char[int(index_char[int(tf.math.argmax(rc(process_img(i,Type='char',is_gray=is_gray))[0]))])]
                if rc_result != "none" : read_char = read_char + str(rc_result) #read_char.append(rc_result)
            except :
                read_char = "rc eror"
                break
        else : 
            rc_result = to_char[int(index_char[int(tf.math.argmax(rc(process_img(i,Type='char',is_gray=is_gray))[0]))])]
            if rc_result != "none" : read_char = read_char + str(rc_result) #read_char.append(rc_result)
    if skip_eror_rp:
        try:
            province = to_province[int(index_province[tf.argmax(rp(process_img(province,is_gray=is_gray,Type='province'))[0])])]
        except:
            province = "rp eror"
    else : province = to_province[int(index_province[tf.argmax(rp(process_img(province,is_gray=is_gray,Type='province'))[0])])]

    return read_char , province

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
#   print(image_center)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

