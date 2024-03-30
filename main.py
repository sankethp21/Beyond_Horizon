import cv2
import sys

sys.path.append('Module-1')
from voice import voice

sys.path.append('Module-2')
from OCR import ocr

sys.path.append('Module-3')
from Image_Captioning import caption_this_image

sys.path.append('Module-4')
from reco import recognise

sys.path.append('Module-5')
from classification import classify

sys.path.append('Module-6')
from traffic_light import traffic_lig

sys.path.append('../')


mode=0
count = 0
def cam():
    global mode
    global count
    cap = cv2.VideoCapture(0)
    while True:
        if count%4==0:
            mode=0
        ret, frame = cap.read()
        img_path="Frames/frame"+str(count)+".jpg"
        print(img_path)
        cv2.imwrite(img_path, frame)
        count+=1
        if mode!=0:
            if(mode==1):
                voice(caption_this_image(img_path))
                continue
            elif(mode==2):
                names=recognise(img_path)
                for name in names:
                    voice(name)
                continue
            elif(mode==3):
                voice(classify(img_path))
                continue
            elif(mode==4):
                i=traffic_lig(img_path)
                if i==0:
                    voice('You must stop, Because traffic light is red')
                elif i==1:
                    voice('Traffic light is Yellow and red light is about to appear')
                elif i==2:
                    voice('You can move forward as the traffic light is green')
                else:
                    voice('Traffic light is off or not detected')
                continue
            elif(mode==5):
                voice(ocr(img_path))
                continue
        voice("enter the key")
        cv2.imshow("frame",frame)
        key = cv2.waitKey(10000)
        if key==-1:
            voice('key is not detected')
            break
        if key == 49:
            voice("Surrounding Description Mode Activated")
            mode=1
        elif key == 50:
            voice('Facial Recognition Mode Activated')
            mode=2
        elif key == 51:
            voice('Road Sign Recognition Mode Activated')
            mode=3
        elif key == 52:
            voice('Traffic Light recognition Mode Activated')
            mode=4
        elif key == 53:
            voice('OCR Mode Activated')
            mode=5
        elif key == 27:
            voice('powering off')
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    cam()
