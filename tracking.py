import cv2
import numpy as np

video_file = 'background_subtraction_output.mp4'
output_path = './track_output.mp4'

VideoSignal = cv2.VideoCapture(video_file)
width = int(VideoSignal.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(VideoSignal.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(VideoSignal.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'DIVX')
output = cv2.VideoWriter(output_path, codec, fps, (width, height)) 

YOLO_net = cv2.dnn.readNet("data/yolov3_mask_last.weights", "data/detect_mask.cfg")
YOLO_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
YOLO_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


classes = []
with open("data/object.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i[0]-1] for i in YOLO_net.getUnconnectedOutLayers()]

while cv2.waitKey(1)<0:
    ret, frame = VideoSignal.read()
    h,w,c = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (512,512),(0,0,0),True,crop=False)
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

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(frame, label, (x,y-20), cv2.FONT_ITALIC, 1, (0,0,255), 1)

    output.write(frame)
    cv2.namedWindow("YOLOv3", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv3", frame)
   
