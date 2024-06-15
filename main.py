import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st
st.set_page_config(layout='wide')
st.title("math problem solver based on Opencv and LLM (google gemini) ✨")
st.write("""This project is based on the media pipe library, open CV, and generative.ai. You can write a math equation on canvas using one finger and send it to the LLM Gemini to receive a response.
        \n🚀How do I interact with the project:
        \n 1- Use your finger to write the math equation.
        \n 2- To send the input to Gemini, raise all the fourth fingers.
        \n 3- To erase the canvas raise all five fingers""")
column1,column2=st.columns([2,1])
with column1:
     #run=st.checkbox("check camera",value=True)
     frameWindow=st.image([])
with column2:
     st.title("Answer")   
     outputText=st.subheader("")

genai.configure(api_key="AIzaSyB6XWyw0G5g_kVoHHHP20mHovf-KiIONQI")
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
# The '2' indicates the third camera connected to your computer; '0' would usually refer to the built-in camera
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    # Continuously get frames from the webcam
    # Find hands in the current frame
    # The 'draw' parameter draws landmarks and hand outlines on the image if set to True
    # The 'flipType' parameter flips the image, making it easier for some detections
    hands, img = detector.findHands(img, draw=False, flipType=True)

        # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
    
        lmList = hand1["lmList"]  # List of 21 landmarks for the first hand
            #bbox1 = hand1["bbox"]  # Bounding box around the first hand (x,y,w,h coordinates)
            #center1 = hand1['center']  # Center coordinates of the first hand
            #handType1 = hand1["type"]  # Type of the first hand ("Left" or "Right")

            # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand1)
        return fingers,lmList
    else:
        return None

def draw(info,previousPosition,canvas):
    fingers, lmlist=info
    currentPosition=None
    if fingers==[0,1,0,0,0]:
          currentPosition=lmlist[8][0:2]
          if previousPosition is None: 
               previousPosition=currentPosition
          cv2.line(canvas,currentPosition,previousPosition,(255,0,255),10)
    elif fingers==[1,1,1,1,1]:
         #reset canvas
         canvas= np.zeros_like(img)     
    return currentPosition,canvas

def sendToGemeni(model,canvas,fingers):
    if fingers==[1,1,1,1,0]:
        Pil_image=Image.fromarray(canvas)
        response = model.generate_content(["solve this math problem",Pil_image])
        #response = model.generate_content("Write a story about a AI and magic")
        return response.text
previousPosition=None
canvas=None
combinedImage=None
outputResult=""
while True:
        # Capture each frame from the webcam
        # 'success' will be True if the frame is successfully captured, 'img' will contain the frame
        success, img = cap.read()
        img=cv2.flip(img,1)
        if canvas is None:
            canvas=np.zeros_like(img)
            combinedImage=img.copy
        info=getHandInfo(img)
        if info:
            fingers,lmlist=info
            
            previousPosition,canvas=draw(info,previousPosition,canvas)
            outputResult=sendToGemeni(model,canvas,fingers)
        combinedImage=cv2.addWeighted(img,0.7,canvas,0.3,0)
              
        # Display the image in a window
        #cv2.imshow("Image", img)
        #cv2.imshow("canvas", canvas)
        #cv2.imshow("combined Image", combinedImage)
        frameWindow.image(combinedImage,channels="BGR")
        if(outputResult):
            outputText.text(outputResult)

        # Keep the window open and update it for each frame; wait for 1 millisecond between frames
        if cv2.waitKey(1)==ord('q'):
             break
cv2.destroyAllWindows()        

