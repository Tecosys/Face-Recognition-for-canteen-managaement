import streamlit as st
import sqlite3
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Database setup
def setup_database():
    conn = sqlite3.connect('food_pref.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS StudentFoodPreferences
                 (rollno TEXT PRIMARY KEY, name TEXT, photo BLOB, food_pref TEXT)''')
    conn.commit()
    conn.close()

# Capture photo
def capture_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame

# Save student data
def save_student(rollno, name, photo, food_pref):
    _, buffer = cv2.imencode('.jpg', photo)
    photo_blob = buffer.tobytes()
    conn = sqlite3.connect('food_pref.db')
    c = conn.cursor()
    c.execute("INSERT INTO StudentFoodPreferences (rollno, name, photo, food_pref) VALUES (?, ?, ?, ?)",
              (rollno, name, photo_blob, food_pref))
    conn.commit()
    conn.close()

# Load data from the database
def load_data():
    conn = sqlite3.connect('food_pref.db')
    c = conn.cursor()
    c.execute("SELECT rollno, name, photo, food_pref FROM StudentFoodPreferences")
    data = c.fetchall()
    conn.close()
    
    photos = []
    labels = []
    for row in data:
        rollno, name, photo, food_pref = row
        nparr = np.frombuffer(photo, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        photos.append(img_np)
        labels.append(1 if food_pref == 'non-veg' else 0)  # Label: 1 for non-veg, 0 for veg

    return np.array(photos), np.array(labels)

# Train model
def train_model():
    photos, labels = load_data()
    photos = photos / 255.0  # Normalize

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(photos, labels, epochs=10)
    model.save('food_pref_model.h5')
    return model

# Recognize and predict food preference
def recognize_and_predict():
    model = tf.keras.models.load_model('food_pref_model.h5')
    input_shape = model.input_shape[1:3]  # Get the required input shape

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            return None, "Failed to capture image"
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, input_shape)  # Adjust size as needed
            face_resized = np.expand_dims(face_resized, axis=0) / 255.0
            
            prediction = model.predict(face_resized)
            food_pref = 'non-veg' if prediction > 0.5 else 'veg'
            
            cap.release()
            return frame, food_pref

        if len(faces) > 0:
            break

    cap.release()
    return None, "No face detected"

# Streamlit UI
setup_database()
st.title("Face Recognition System")
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f0;
    }
    .stButton>button {
        width: 100%;
        margin-top: 10px;
        margin-bottom: 10px;
        padding: 10px;
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
    }
    .stTextInput>div>div>input {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    .stSelectbox>div>div>div>div>div {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Register new student
st.header("Register New Student")
rollno = st.text_input("Enter Roll Number")
name = st.text_input("Enter Name")
food_pref = st.selectbox("Enter Food Preference", ["veg", "non-veg"])
if st.button("Capture & Register"):
    photo = capture_photo()
    if photo is not None:
        save_student(rollno, name, photo, food_pref)
        st.image(photo, channels="BGR", caption="Captured Photo", use_column_width=True)
        st.success("Student registered successfully!")
    else:
        st.error("Failed to capture photo.")

# Train model
st.header("Train Model")
if st.button("Train Model"):
    with st.spinner('Training model...'):
        model = train_model()
    st.success("Model trained successfully!")

# Recognize and predict
st.header("Recognize and Predict Food Preference")
if st.button("Start Recognition"):
    frame, food_pref = recognize_and_predict()
    if frame is not None:
        st.image(frame, channels="BGR", caption=f"Detected Face: {food_pref}", use_column_width=True)
        st.success(f"Food Preference: {food_pref}")
    else:
        st.error(food_pref)
