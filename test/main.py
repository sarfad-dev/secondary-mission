import cv2
import numpy as np
import time
import os
from sqlalchemy import create_engine, Column, Float, Integer, MetaData, Table
from sqlalchemy.orm import registry, sessionmaker

mapper_registry = registry()

engine = create_engine('mysql+pymysql://root:@0.tcp.eu.ngrok.io:15383/sarfad')
mapper_registry.map_imperatively

metadata = MetaData()

speed_data = Table('SpeedData', metadata,
                   Column('id', Integer, primary_key=True),
                   Column('Time', Float),
                   Column('Speed', Float))

metadata.create_all(engine)

class SpeedData(object):
    def __init__(self, Time, Speed):
        self.Time = Time
        self.Speed = Speed

mapper_registry.map_imperatively(SpeedData, speed_data)

Session = sessionmaker(bind=engine)
session = Session()

cap = cv2.VideoCapture('video.mp4')

if not cap.isOpened():
    print("Error: Unable to open video file")
    exit()

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

distance_traveled = 0
start_time = time.time()
previous_time = start_time
frame_count = 0

speed_interval = 5  

scale_factor = 10 / 10.2

def smooth_speed(speeds, window_size=5):
    smoothed_speeds = np.convolve(speeds, np.ones(window_size)/window_size, mode='valid')
    return smoothed_speeds

speeds = []

try:
    while(cap.isOpened()):
        ret, frame2 = cap.read()
        if not ret:
            break
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)

        avg_motion = np.mean(mag)

        distance_traveled += avg_motion

        prvs = next_frame
        frame_count += 1

        current_time = time.time()
        elapsed_time = current_time - previous_time

        if elapsed_time >= speed_interval:
            speed_mps = round(distance_traveled / speed_interval / scale_factor, 2)

            print(f"Instantaneous speed at {round(frame_count / 60, 2)} seconds:", speed_mps, "meters per second")


            cv2.imwrite(f'images/frame_{str(round(frame_count / 60, 2)).replace(".", ",")}.jpg', frame2)

            speed_entry = SpeedData(Time=round(frame_count / 60, 2), Speed=speed_mps)
            session.add(speed_entry)
            session.commit()

            speeds.append(speed_mps) 

            distance_traveled = 0
            previous_time = current_time

            scale_factor += 0.0001

finally:
    speeds = smooth_speed(speeds)  
    session.close()
    cap.release()
