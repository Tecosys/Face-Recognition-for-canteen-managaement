Face Recognition System with Food Preference Prediction

This project captures student photos via webcam, stores them in an SQLite database, and uses a neural network model to recognize faces and predict 
food preferences (veg/non-veg) in real-time. It leverages OpenCV for face detection and TensorFlow for prediction. The user interface is built using Streamlit.

Features

•	Register New Student: Capture student photos and save them along with roll number, name, and food preference to an SQLite database.

•	Train Model: Load student data from the database and train a neural network model to predict food preferences.

•	Recognize and Predict: Capture a photo in real-time, recognize the student's face, and predict their food preference.

Installation

1.	Clone the repository:
   
    git clone https://github.com/yourusername/face-recognition-food-pref.git

    cd face-recognition-food-pref

2.	Install the required packages:
   
    pip install -r requirements.txt

3.	Run the application:
   
    streamlit run app.py

Usage

1.	Register a new student:
   
    o	 Enter the roll number, name, and food preference (veg/non-veg).

    o	 Capture the student's photo using the webcam.

    o	 The captured photo and student details are stored in the database.

2.	Train the model:
   
     o	Click the "Train Model" button to train the neural network model using the stored student data.

     o	The trained model is saved to disk.


3.	Recognize and predict food preference:
   
    o	 Click the "Start Recognition" button to capture a photo in real-time.

    o	 The system detects the face, recognizes the student, and predicts their food preference.

    o	 The predicted food preference is displayed along with the captured photo.

Technologies Used

•	Python: Programming language used for implementation.

•	Streamlit: Framework for creating the web-based user interface.

•	OpenCV: Library for real-time computer vision tasks, including face detection and image processing.

•	TensorFlow: Library for training and deploying the neural network model.

•	SQLite: Database for storing student information and photos.

Project Structure

•	app.py: Main application script.

•	requirements.txt: List of required packages.

•	food_pref_model.h5: Trained neural network model for food preference prediction (generated after training)

Note

•	Ensure that your webcam is connected and functional.

•	The project requires cv2, numpy, tensorflow, streamlit, and PIL libraries to be installed.

•	The model is trained using images of size 480x640. Adjust the image size in the code if needed.

