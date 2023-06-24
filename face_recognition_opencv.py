# -*- coding: utf-8 -*-
"""
Face Recognition Using OpenCV

This script uses Python's face_recognition library, which has a model with 99.38% accuracy in detecting faces and their shapes/formats.
"""

# Import necessary libraries
import os
import cv2
import face_recognition
import matplotlib.pyplot as plt
from IPython.display import Video,display

# Define function to encode all faces
def encode_all_faces(folder_path) -> list:
    """Function to return a list of all encodings."""

    file_list = os.listdir(folder_path)
    encodings = []
    names = []

    for file_name in file_list:
        img = cv2.imread(os.path.join(folder_path, file_name))
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_encoding = face_recognition.face_encodings(rgb_img)
        if img_encoding:
            encodings.append(img_encoding[0])
            name = os.path.splitext(file_name)[0].replace("_", " ")
            names.append(name)

    return encodings, names

# Define function to find matching image
def find_matching_image(names, encodings, img_encoding, folder_path):
    """Function to find matching image."""

    img_idx = -1
    for i, encoding in enumerate(encodings):
        matches = face_recognition.compare_faces([encoding], img_encoding)

        if True in matches:
            img_idx = i
            name = names[i]
            break

    if img_idx == -1:
        return "No matching image found"
    else:
        file_list = os.listdir(folder_path)
        matching_image_path = os.path.join(folder_path, file_list[img_idx])
        img = cv2.imread(matching_image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.imshow(img_rgb)
        plt.axis('off')
        plt.legend("Matched Image: " + name)
        plt.show()

        return name, matching_image_path

# Define function to create a VideoWriter object
def video_writer(video_capture, output_file):
    """Function to write video."""

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# Define function for face recognition in videos
def video_recognition(video_capture, encodings, names, output_video):
    """Function to recognize faces in video."""

    frame_count = 0  # frame number for debug purposes

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        print("Frame: ", frame_count)

        if not ret:
            break
        # Resize the frame
        scale_percent = 20  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            matches = face_recognition.compare_faces(encodings, face_encoding)

            if True in matches:
                name = names[matches.index(True)]
                #print(name)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            output_video.write(frame)
        
        cv2.imshow('Video', frame)  # This line will display the current frame in a window named 'Video'
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
          
        frame_count += 1

    video_capture.release()
    output_video.release()

    video = Video(output_video.filename, embed=True)
    display(video)
    cv2.destroyAllWindows()
def webcam_recognition(video_capture,encodings, names):
    """Function to recognize faces from webcam."""

    
    frame_count = 0  # frame number for debug purposes

    while True: # infinite loop to continuously capture from the webcam
        ret, frame = video_capture.read()
        print("Frame: ", frame_count)

        if not ret:
            break
        # Resize the frame
        scale_percent = 100  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"

            matches = face_recognition.compare_faces(encodings, face_encoding)

            if True in matches:
                name = names[matches.index(True)]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
        cv2.imshow('Webcam Video', frame)  # This line will display the current frame in a window named 'Video'
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
          
        frame_count += 1

    video_capture.release()
    cv2.destroyAllWindows()


# Main program
if __name__ == "__main__":
    current_path = os.getcwd()

    # Encode all faces
    folder_path = os.path.join(current_path, 'images')
    encodings, names = encode_all_faces(folder_path)
  
    flag = 'w'
      
    if flag == 'i':
      # Image recognition
      test_image_path = "Messi1.webp"
      test_img = cv2.imread(test_image_path)
      rgb_img_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
      plt.imshow(rgb_img_test)
      plt.axis('off')
      plt.legend("Test Image: ")
      plt.show()
      img_encoding_test = face_recognition.face_encodings(rgb_img_test)[0]
      find_matching_image(names, encodings, img_encoding_test, folder_path)

    elif flag == 'v':
        # Video recognition
    # Face recognition in downloaded videos
      video_path = os.path.join(current_path, "videos")
      video_file = os.path.join(video_path, "messi video.mp4")
      output_video_file = "/output_video.mp4"
      video_capture = cv2.VideoCapture(video_file)
      output_video = video_writer(video_capture, output_video_file)
      video_recognition(video_capture, encodings, names, output_video)

    elif flag == 'w':
        # Webcam recognition
        video_capture = cv2.VideoCapture(0)
        webcam_recognition(video_capture, encodings, names)
