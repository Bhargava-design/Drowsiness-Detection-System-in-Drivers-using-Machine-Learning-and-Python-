import cv2
import time
import tkinter as tk
import winsound
import os
import numpy as np
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the cap variable globally
cap = None

#Load pre-trained face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

#Define the path for non drowsy and drowsy
drowsy_path = r"C:/Users/gudur/Desktop/Project1/Drowsy Images/"
non_drowsy_path = r"C:/Users/gudur/Desktop/Project1/Non Drowsy Images/"

#start_camera
def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open the camera")
        return
    
    run_camera_loop()

#stop_camera
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
        
#Function to capture and save the images
def capture_and_save_image(frame, drowsy):
    folder_path = drowsy_path if drowsy else non_drowsy_path
    filename = f"{folder_path}image_{int(time.time())}.jpg"
    cv2.imwrite(filename, frame)
    print("Image captured:", filename)
    if drowsy:
        winsound.Beep(500, 1000)
        
    
#reload images
def reload_images(folder_path, label, target_shape):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, target_shape)
            images.append(img_resized)
            labels.append(label)
    return images, labels

#run camera loop
def run_camera_loop():
    global cap
    if cap is None or not cap.isOpened():
        print("Error: Camera is not Opened")
        return
    
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x,y, W, h) in faces:
            cv2.rectangle(frame, (x,y), (x+W, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+W]
            roi_color = frame[y:y+h, x:x+W]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            drowsy = detect_drowsiness(roi_color, eyes)
            capture_and_save_image(roi_color, drowsy)
            
            if drowsy:
                cv2.putText(frame, "Drowsy", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Non Drowsy", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) == 27: #Exit if the kay is pressed
        stop_camera()
        return
    #call the function again after a delay
    root.after(10, run_camera_loop)
    
# detection function
def detect_drowsiness(frame, eyes):
    if len(eyes) == 0:
        return True #Drowsy
    return False #Non Drowsy

#Load and pre[processing method
def load_and_preprocess(folder_path, label, common_shape):
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, target_shape)
            images.append(img_resized)
            labels.append(label)
    return images, labels

# CNN Model
def cnn_model(drowsy_images, non_drowsy_images, drowsy_labels, non_drowsy_labels):
    #Combine both drowsy and non-drowsy images and labels
    X = np.array(drowsy_images + non_drowsy_images)
    y = np.concatenate([drowsy_labels, non_drowsy_labels])
    
    print("X:",X)
    print("y:", y)
    
    #Equalize the number of smaples if arrays have different sizes
    min_size = min(len(X), len(y))
    X = X[:min_size]
    y = y[:min_size]
    
    print("Equalized X:", X)
    print("Equalized y:", y)
    
    #Normalize the pixel values to range  [0, 1]
    X = X/255.0
    
    #Reshape images to match CNN imput shape (add channel dimension)
    X =X.reshape(-1, X.shape[1], X.shape[2],1)
    
    # Split the date into training and testing sets
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
    
    print("X train for CNN Model: ", X_train)
    print("X test for CNN Model: ", X_test)
    print("y train for CNN Model: ", y_train)
    print("y test for CNN Model: ", y_test)
    
    #CNN model Architecture implementation
    cnn_model = Sequential([
         Conv2D(4, (2, 2), input_shape=(X_train.shape[1], X_train.shape[2], 1), activation = 'relu'),
         MaxPooling2D(pool_size = (2, 2)),
         Conv2D(4, (2, 2), activation = 'relu'),
         MaxPooling2D(pool_size = (2, 2)),
         Conv2D(4, (2, 2), activation = 'relu'),
         MaxPooling2D(pool_size = (2, 2)),
         Conv2D(4, (2, 2), activation = 'relu'),
         MaxPooling2D(pool_size = (2, 2)),
         Flatten(),
         Dense(32, activation = 'relu'),
         Dense(1, activation = 'relu')
    ])
    # complie the CNN Model
    cnn_model.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)
    
    #loss and accuarcy of this model
    loss, accuracy = cnn_model.evaluate(X_test, y_test)
    print("CNN Accuracy:", accuracy)

# KNN algorithm
def KNN_Model():
    #Define some common shape 
    common_shape = (100, 100)
    
    #Load and preprocessing
    drowsy_images, drowsy_labels = load_and_preprocess(drowsy_path, 1, common_shape)
    non_drowsy_images, non_drowsy_labels = load_and_preprocess(non_drowsy_path, 0, common_shape)
    
    #Concatenate the arrays
    X_drowsy = np.array(drowsy_images)
    X_non_drowsy_images = np.array(non_drowsy_images)
    X = np.concatenate((X_drowsy,X_non_drowsy_images), axis = 0)
    y = np.concatenate([drowsy_labels,non_drowsy_labels])
    
    #Flatten the images
    X_Flat = np.array([img.flatten() for img in X])
    #Split and train and test
    X_train, X_test, y_train, y_test = train_test_split(X_Flat, y, test_size=0.2, random_state = 42)
    print("X train for CNN Model: ", X_train)
    print("X test for CNN Model: ", X_test)
    print("y train for CNN Model: ", y_train)
    print("y test for CNN Model: ", y_test)
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_accuarcy = accuracy_score(y_test, knn_pred)
    print("KNN Accuracy:", knn_accuarcy)
    
# Random Foest
def Random_Forest():
    #Define some common shape 
    common_shape = (100, 100)
    
    #Load and preprocessing
    drowsy_images, drowsy_labels = load_and_preprocess(drowsy_path, 1, common_shape)
    non_drowsy_images, non_drowsy_labels = load_and_preprocess(non_drowsy_path, 0, common_shape)
    
    #Concatenate the arrays
    X_drowsy = np.array(drowsy_images)
    X_non_drowsy_images = np.array(non_drowsy_images)
    X = np.concatenate((X_drowsy,X_non_drowsy_images), axis = 0)
    y = np.concatenate([drowsy_labels,non_drowsy_labels])
    
    #Flatten the images
    X_Flat = np.array([img.flatten() for img in X])
    #Split and train and test
    X_train, X_test, y_train, y_test = train_test_split(X_Flat, y, test_size=0.2, random_state = 42)
    print("X train for CNN Model: ", X_train)
    print("X test for CNN Model: ", X_test)
    print("y train for CNN Model: ", y_train)
    print("y test for CNN Model: ", y_test)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuarcy = accuracy_score(y_test, rf_pred)
    print("Random Forest Accuracy:", rf_accuarcy)
    
#main function
if __name__ == "__main__":
    #initialize the tkinter
    root = tk.Tk()
    root.title("Drowsiness Detection")
    
    #create a frame
    frame = tk.Frame(root)
    frame.pack(padx=60, pady=60)
    
    #create a label to display the camera feed
    label = tk.Label(frame)
    label.pack()
    
    #create buttons
    open_camera_button = tk.Button(frame, text = "Open Camera", command=start_camera)
    open_camera_button.pack(side=tk.LEFT, padx=10)
    
    stop_button = tk.Button(frame, text = "Close Camera", command=stop_camera)
    stop_button.pack(side=tk.LEFT, padx=20)
    
    root.mainloop()
    
    target_shape = (100, 100) #desired shape
    
    drowsy_images, drowsy_labels = reload_images(drowsy_path, 1, target_shape)
    non_drowsy_images, non_drowsy_labels = reload_images(non_drowsy_path, 0, target_shape)
    cnn_model(drowsy_images, non_drowsy_images, drowsy_labels, non_drowsy_labels)
    KNN_Model()
    Random_Forest()
    
    

    
    
    
     
