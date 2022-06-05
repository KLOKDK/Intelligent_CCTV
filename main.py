import cv2
import numpy as np

# For Report
def padding(img, width, height):
    delta_w = width - img.shape[1]
    delta_h = height - img.shape[0]
    result = cv2.copyMakeBorder(img,0,delta_h,0,delta_w,cv2.BORDER_CONSTANT)
    return result


video_path = 'test_Trim.mp4'
output_path = './background_subtraction_output.mp4'

video = cv2.VideoCapture(video_path)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'DIVX')
output_video = cv2.VideoWriter(output_path, codec, fps, (540,360)) 

# 배경과 객체 구분을 위한 함수
fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=200,nmixtures=3,backgroundRatio=0.7, noiseSigma=0)
#fgbg = cv2.createBackgroundSubtractorMOG2(history=200,varThreshold=16,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=20, decisionThreshold=0.5)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,1))
close_kernel = np.ones((5,5), np.uint8)
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))

# YOLO 사용
YOLO_net = cv2.dnn.readNet("yolov3/yolov3_mask_last.weights", "yolov3/detect_mask.cfg")
YOLO_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
YOLO_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = []
COLORS = [[0,255,0],[0,0,0],[0,0,255]]

with open("yolov3/object.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in YOLO_net.getUnconnectedOutLayers()]
ptz_loc = np.zeros((2,2))
tracking_frame = np.zeros((940,540))

if video.isOpened():
    rv, frame = video.read()
    print("Frame Size:",frame.shape[1],frame.shape[0])
    while cv2.waitKey(1)<0:        
        return_value, frame = video.read()
        
        # 프레임 정보가 있으면 계속 진행 
        if return_value:
            pass
        else : 
            print('비디오가 끝났거나 오류가 있습니다')
            break
        frame = cv2.resize(frame,(1080, 720))
        h,w,c = frame.shape
        
        # 배경 제거 마스크
        background_extraction_mask = fgbg.apply(frame)
        background_extraction_mask = cv2.morphologyEx(background_extraction_mask, cv2.MORPH_OPEN, kernel)
        background_extraction_mask = cv2.morphologyEx(background_extraction_mask, cv2.MORPH_CLOSE, kernel)
        #background_extraction_mask = cv2.dilate(background_extraction_mask,kernel,iterations=1)
        background_extraction_mask = np.stack((background_extraction_mask,)*3, axis=-1)
        background_inverse = cv2.bitwise_not(background_extraction_mask)

        # 배경이 제거된 프레임
        bitwise_image = cv2.bitwise_and(frame, background_extraction_mask)
        object_image = bitwise_image.copy()

        # 배경만 존재하는 프레임
        background_image = cv2.bitwise_and(frame, background_inverse)

        background_extraction_output = bitwise_image.copy()
        background_concat_image = np.concatenate((background_inverse, background_image), axis=1)
        
        # Mask Detection
        blob = cv2.dnn.blobFromImage(bitwise_image, 1 / 255.0, (864, 864),(0,0,0),True,crop=False)
        YOLO_net.setInput(blob)
        outs = YOLO_net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    dw = int(detection[2] * w)
                    dh = int(detection[3] * h)

                    x = int(center_x - dw/2)
                    y = int(center_y - dh/2)
                    boxes.append([x,y,dw,dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]
                color = [int(c) for c in COLORS[class_ids[i]]]

                cv2.rectangle(bitwise_image, (x,y), (x+w,y+h), color, 2)
                cv2.putText(bitwise_image, label, (x,y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

                if label == 'no mask':
                    print("No Mask location:", x, y)                   
                    ptz_loc[0][0] = x - 150
                    ptz_loc[0][1] = x +  150
                    ptz_loc[1][0] = y - 50
                    ptz_loc[1][1] = y + 200

                    ptz_loc[0] = np.clip(ptz_loc[0],0,bitwise_image.shape[1])
                    ptz_loc[1] = np.clip(ptz_loc[1],0,bitwise_image.shape[0])

        result = bitwise_image + background_image

        location = ptz_loc.astype(np.uint32)
        tracking_frame = frame[location[1][0]:location[1][1],location[0][0]:location[0][1]]
        ptz = padding(tracking_frame,300,250)
        

        # For Report
        third_camera = frame[360:,:540]
        third_bitwise_image = bitwise_image[360:,:540]
        background_extraction_output = background_extraction_output[360:,:540]
        object_image = object_image[360:,:540]
  
        #cv2.imshow('result',result)
        #cv2.imshow('ptz',ptz)
        #cv2.imshow('3 camera', third_camera)
        #cv2.imshow('With YOLO', object_image)
        cv2.imshow('with YOLO', third_bitwise_image)
        output_video.write(third_bitwise_image)

video.release()
cv2.destroyAllWindows()