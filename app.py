import cv2
import numpy as np
import os
from flask import Flask, Response, jsonify, render_template, request
from demo import FER_on_image
import sys
import random
import subprocess

app = Flask(__name__)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the folder path where captured frames will be stored
captured_frames_folder = 'captured_frames'

# Create the folder if it doesn't exist
os.makedirs(captured_frames_folder, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    selected_option = request.json.get("selectedOption")
    print("Selected Option:", selected_option, file=sys.stderr)

    folder_mapping = {
        "Track 1": "track1",
        "Track 2": "track2"
    }

    success, frame = cap.read()

    if success:
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        if ret:
            # Save the frame as an image in the 'captured_frames' folder
            frame_number = len(os.listdir(captured_frames_folder))
            frame_filename = os.path.join(captured_frames_folder, f'Image_{frame_number:06d}.jpg')
            cv2.imwrite(frame_filename, frame)

            # Log a success message to the console
            print('Image saved successfully.')

            # Predict emotion on the captured frame
            pred = FER_on_image(frame_filename)

            # Select and play a random song based on the selected option
            if selected_option in folder_mapping:
                selected_folder = folder_mapping[selected_option]
                songs = os.listdir(selected_folder)
                random_song = random.choice(songs)
                song_path = os.path.join(selected_folder, random_song)

                os.startfile(song_path)

                # Return the prediction and song path to the client
                return jsonify({'success': True, 'emotion': pred, 'song_path': song_path})
            else:
                return jsonify({'success': False, 'error': 'Invalid option'})

    return jsonify({'success': False, 'error': 'Failed to capture and save frame'})

if __name__ == '__main__':
    app.run(debug=True)
