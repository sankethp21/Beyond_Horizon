from PIL import Image
import pytesseract
import numpy as np

def textRecognition(filename):
  img1 = np.array(Image.open(filename))
  text = pytesseract.image_to_string(img1)
  l=''.join(text)
  l=l.split('\n')
  li=[]
  for s in l:
    s1="".join(c for c in s if c.isalpha() or c.isdigit() or c==" " or c in "!@#$%&*.?':;_" )
    if s1!="":
      li.append(s1) 
  temp=" ".join(li)
  temp=temp.split('.')
  return temp
  
def ocr(img):
  pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
  sentences=textRecognition(img)
  for s in sentences:
    count=sum(len(x) for x in s.split())
    print("length:")
    print(count)
    if count<=2: continue
    return s
