import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import keras 
from keras.models import model_from_json
import time

# Initialize Firebase with your service account credentials
cred = credentials.Certificate('FireBaseKey.json')
firebase_admin.initialize_app(cred, {
     'databaseURL':'https://iot-project-cows-default-rtdb.firebaseio.com/'
})

# Define a function to upload data to Firebase
def upload_to_firebase(name, age, emotion):
    # Create a reference to the 'emotions' path in the Realtime Database
    ref = db.reference('emotions')

    # Generate a unique key based on time for FIFO order
    unique_key = str(int(time.time() * 1000))

    # Set data at the 'emotions' path with the unique key
    ref.child(unique_key).set({
        'name': name,
        'age': age,
        'emotion': emotion
    })

# Define a function to extract features from an image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature/255.0

# Load the trained facial emotion model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

model.load_weights("facialemotionmodel.h5")

# Load the Haar cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Create a dictionary to map labels to emotion names
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Start the webcam
webcam = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    i, im = webcam.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    # Process each detected face
    for (p, q, r, s) in faces:
        # Extract the face region from the grayscale frame
        image = gray[q:q+s, p:p+r]

        # Draw a rectangle around the face
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)

        # Resize the face region to 48x48 pixels
        image = cv2.resize(image, (48, 48))

        # Extract features from the resized face region
        img = extract_features(image)

        # Make a prediction using the trained facial emotion model
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]

        # Prompt the user for their name and age
        name = input("Enter your name: ")
        age = input("Enter your age: ")

        # Upload the detected emotion and user information to Firebase
        upload_to_firebase(name, age, prediction_label)

        # Display the predicted emotion on the frame
        cv2.putText(im, '%s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

    # Display the webcam frame
    cv2.imshow("Output", im)

    # Check for a 'q' key press to quit
    if cv2.waitKey(27) == ord('q'):
        break

# Close the webcam
webcam.release()

# Close all windows
cv2.destroyAllWindows()
