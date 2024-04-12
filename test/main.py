import cv2
import numpy as np
import time
import csv

# Load the video file
cap = cv2.VideoCapture('video.mp4')
if not cap.isOpened():
    print("Error: Unable to open video file")
    exit()r
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Variables for speed calculation
distance_traveled = 0
start_time = time.time()
previous_time = start_time
frame_count = 0

# Interval for calculating speed (in seconds)
speed_interval = 5  # Adjust as needed

# Determine the scale factor (pixels per meter) based on the expected speed
# For example, if expected speed is 7 m/s and corresponding pixel distance is 100 pixels
# then scale factor = 100 pixels / 7 meters
scale_factor = 10 / 7  # Example scale factor, replace with your actual value

# Open CSV file for writing
with open('values.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (seconds)','Speed (m/s)'])

    while(cap.isOpened()):
        ret, frame2 = cap.read()
        if not ret:
            break
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow using Lucas-Kanade method
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 5, 15, 3, 5, 1.2, 0)

        # Calculate displacement
        dx = np.mean(flow[:,:,0])
        dy = np.mean(flow[:,:,1])

        # Update distance traveled
        distance_traveled += np.sqrt(dx**2 + dy**2)

        # Update previous frame
        prvs = next_frame
        frame_count += 1

        # Calculate elapsed time
        current_time = time.time()
        elapsed_time = current_time - previous_time

        # Calculate speed at specified interval
        if elapsed_time >= speed_interval:
            # Convert speed from pixels per second to meters per second
            speed_mps = round(distance_traveled / elapsed_time / scale_factor, 2)

            # Write time and speed to CSV file
            writer.writerow([frame_count / 60, speed_mps])

            print(f"Instantaneous speed at {round(frame_count / 60,2)} seconds:", speed_mps, "meters per second")

            # Reset distance traveled and update previous time
            distance_traveled = 0
            previous_time = current_time

# Release the capture
cap.release()
