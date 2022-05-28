import cv2
import time
import numpy as np

video_path = 'liverpool.mp4'
output_path = './background_subtraction_output.mp4'

video = cv2.VideoCapture(video_path)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))
codec = cv2.VideoWriter_fourcc(*'DIVX')

# codec = int(video.get(cv2.CAP_PROP_FOURCC))
print(codec, type(codec))

# # # codec = chr(codec&0xff) + chr((codec>>8)&0xff) + chr((codec>>16)&0xff) + chr((codec>>24)&0xff) 
# # # codec = cv2.VideoWriter_fourcc(*codec)
# # print(codec)
out = cv2.VideoWriter(output_path, codec, fps, (1200, 900)) 

#fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=200,nmixtures=3,backgroundRatio=0.7, noiseSigma=0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=800,varThreshold=10,detectShadows=False)
#fgbg = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=20, decisionThreshold=0.5)


# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
time_cost = 0
fps_cost = 0
frame_num = 0
rv, ref_frame = video.read()

while(1):
    start = time.time()
    frame_num +=1 
    
    return_value, frame = video.read()

    # 비디오 프레임 정보가 있으면 계속 진행 
    if return_value:
        pass
    else : 
        print('비디오가 끝났거나 오류가 있습니다')
        break

    frame = cv2.resize(frame,(1200,900))
    
    background_extraction_mask = fgbg.apply(frame)
    background_extraction_mask = cv2.morphologyEx(background_extraction_mask, cv2.MORPH_CLOSE, kernel)
    #background_extraction_mask = cv2.dilate(background_extraction_mask,kernel,iterations=1)

    background_extraction_mask = np.stack((background_extraction_mask,)*3, axis=-1)
    bitwise_image = cv2.bitwise_and(frame, background_extraction_mask)

    concat_image = np.concatenate((frame,bitwise_image), axis=1)

    time_temp = 1000*(time.time()-start)  
    if time.time()-start != 0.0:
        fps_temp = 1/(time.time()-start)  
    else:
        fps_temp=0.01

    time_cost = time_cost + time_temp
    fps_cost = fps_cost + fps_temp
    print('소요 시간 : {:.2f} ms \t 평균FPS : {:.2f}'.format(time_temp,fps_temp))

    cv2.imshow('background extraction video', concat_image)
    out.write(bitwise_image)
#    cv2.waitKey(0)
    if cv2.waitKey(1)>0:
        break

print('소요시간 평균 : {:.2f} ms\t 평균FPS : {:.2f}'.format(time_cost / frame_num, fps_cost/ frame_num))
video.release()
cv2.destroyAllWindows()