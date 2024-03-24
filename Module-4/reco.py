import cv2
from simple_facerec import SimpleFacerec

sfr = SimpleFacerec()
sfr.load_encoding_images("Module-4/images/")

def recognise(frame):
    frame = cv2.imread(frame, cv2.IMREAD_COLOR)
    face_locations, names = sfr.detect_known_faces(frame)
    return names
