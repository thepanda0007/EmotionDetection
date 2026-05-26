from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from tensorflow import lite as tflite

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# load tflite model once at startup
interpreter = tflite.Interpreter(model_path="emotion_model.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ['angry', 'happy', 'neutral', 'sad']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion(face):
    interpreter.set_tensor(input_details[0]['index'], face.astype(np.float32))
    interpreter.invoke()
    result = interpreter.get_tensor(output_details[0]['index'])
    return result

def preprocess(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = img_gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face / 255.0
    face = np.expand_dims(face, axis=0)   # (1, 48, 48)
    face = np.expand_dims(face, axis=-1)  # (1, 48, 48, 1)
    return face

@app.post("/predict")
async def predict(file: UploadFile):
    image_bytes = await file.read()
    face = preprocess(image_bytes)

    if face is None:
        return {"mood": "none", "message": "No face detected"}

    result = predict_emotion(face)
    mood = labels[np.argmax(result)]
    confidence = round(float(np.max(result)) * 100, 2)

    return {"mood": mood, "confidence": confidence}