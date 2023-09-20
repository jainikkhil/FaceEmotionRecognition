import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
from deepface import DeepFace

# Load the Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video file
video_path = "check.mp4"  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Set the window size
window_width = 800
window_height = 600
cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Emotion Detection", window_width, window_height)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break  # Break the loop when the video ends
    
    # Convert the frame to grayscale for face detection
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_classifier.detectMultiScale(frame_gray)
    
    emotions_percentages = []  # Reset the emotion percentages for each frame
    
    for (x, y, w, h) in faces:
        # Crop the face region for emotion detection
        face_region = frame[y:y+h, x:x+w]
        
        # Analyze the emotion in the cropped face region
        responses = DeepFace.analyze(face_region, actions=["emotion"], enforce_detection=False)
        
        # Check if there are any responses in the list
        if responses:
            # Get the first response from the list
            first_response = responses[0]
            
            # Get the emotion percentages for all emotions
            emotion_percentages = first_response["emotion"]
            
            # Append the emotion percentages to the list
            emotions_percentages.append(emotion_percentages)
            
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), thickness=3)
    
    # Calculate the cumulative percentages for all emotions
    if emotions_percentages:
        cumulative_percentages = {emotion: sum([percentage[emotion] for percentage in emotions_percentages]) / len(emotions_percentages) for emotion in emotions_percentages[0]}
        
        # Limit the cumulative percentages to 100%
        for emotion in cumulative_percentages:
            cumulative_percentages[emotion] = min(cumulative_percentages[emotion], 100.0)
    else:
        cumulative_percentages = {}
    
    # Display the cumulative percentages as the title
    title_text = "Cumulative Emotion: " + ", ".join([f"{emotion} ({percentage:.2f}%)" for emotion, percentage in cumulative_percentages.items()])
    cv2.setWindowTitle("Emotion Detection", title_text)
    
    # Display the video frame
    cv2.imshow("Emotion Detection", frame)
    
    # Exit the loop if the 'Esc' key is pressed
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()



