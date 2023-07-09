import cv2

def local_plate(image_name:str, input_form="PATH"):
    if input_form == "PATH":
        image = cv2.imread(image_name) # hard [4 40 44 38(stuck) 27 23]
    elif input_form == "IMG":
        image = image_name
    image = cv2.resize(image,[500,500])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=2, sigmaY=2, borderType = cv2.BORDER_DEFAULT)
    canny = cv2.Canny(blur, 60, 100)
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Iterate thorugh contours and draw rectangles around contours
    print(f"num of box : {len(cnts)}")
    count = 0
    keep_box_coor = []
    keep_crop_image = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        if (w>h) and (1.1<(w/h)<2.2) and (40<h<190):  # set filter box coor
            count += 1
            # cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            keep_box_coor.append([x,y,w,h])
            keep_crop_image.append(gray[y:y+h,x:x+w])


    print(f"number of box that found : {count}")
    # cv2.imshow('canny', canny)
    # cv2.imshow('image', image)
    # cv2.imshow("blur",blur)
    # cv2.waitKey(0)
    return keep_box_coor , keep_crop_image