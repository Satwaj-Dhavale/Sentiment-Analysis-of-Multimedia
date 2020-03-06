import cv2
import face_recognition
import time
import predict_model
import os
import movie_p1
import re

# This file is used to generate the output file.

def create_webcam_output():
    video_capture = cv2.VideoCapture(0)
    face_locations = []

    start_time = time.time()

    while True:
        ret, frame = video_capture.read()

        # Reducing frame to 1/2 of its size. Used for faster execution
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        rgb_small_frame = small_frame[:, :, ::-1]
        # Only process every other frame of video to save time
        if time.time() - start_time >= 0.1:  # Check if 1 sec has passed
            # Find all the faces and face encodings in the current frame of video

            face_locations = face_recognition.face_locations(rgb_small_frame)
            print(face_locations)
            start_time = time.time()

        # Display the results
        for (top, right, bottom, left) in face_locations[:2]:
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            font = cv2.FONT_HERSHEY_DUPLEX

            pred = predict_model.prediction(frame[top:bottom, left:right][:, :, ::-1])
            cv2.putText(frame,  pred, (left + 6, bottom + 6), font, 1.5, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


def create_image_output(filepaths = './test_data/test_image.jpg'):
    count = 0
    for filename in filepaths:
        image = cv2.imread(filename)
        face_locations = face_recognition.face_locations(image)
        print(face_locations)

        for (top, right, bottom, left) in face_locations:
            # Draw a box around the face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            # cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255))
            font = cv2.FONT_HERSHEY_DUPLEX

            pred = predict_model.prediction(image[top:bottom, left:right][:, :, ::-1])
            cv2.putText(image, pred, (left + 6, bottom + 6), font, 1.0, (255, 255, 255), 1)

            img_path = r'.\test_data\output\output_' + str(count) + '.jpg'
            cv2.imwrite(img_path, image)
            os.startfile(img_path)
            count += 1
    print("Output Images created")


def create_video_output(File_path = './video_data/test_video.mp4'):
    frame_path = './video_data/frames/'
    video_file = File_path[0]
    out_file = r'.\video_data\output\output_' + video_file.split('/')[-1]

    movie_p1.split_into_frames(frame_path, video_file, fps=10)

    files = [f for f in os.listdir(frame_path) if os.path.isfile(os.path.join(frame_path, f))]

    # For sorting the file by name
    files.sort(key=lambda f: int(re.sub('\D', '', f)))

    for i in files:
        image = cv2.imread(os.path.join(frame_path, i))
        small_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_image)
        print(face_locations)

        for (top, right, bottom, left) in face_locations:
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            font = cv2.FONT_HERSHEY_DUPLEX

            pred = predict_model.prediction(image[top:bottom, left:right][:, :, ::-1])
            cv2.putText(image, pred, (left + 6, bottom + 6), font, 1.0, (255, 255, 255), 1)

            cv2.imwrite(os.path.join(frame_path, i), image)

    movie_p1.convert_frames_to_video(frame_path, out_file, fps=5.0)
    print("Output Video created")
    os.startfile(out_file)
