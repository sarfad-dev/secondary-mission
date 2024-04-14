# Importování potřebných knihoven
import cv2
import numpy as np
import time
import os
from sqlalchemy import create_engine, Column, Float, Integer, MetaData, Table
from sqlalchemy.orm import registry, sessionmaker

# Inicializace registru mapperu pro SQLAlchemy
mapper_registry = registry()

# Připojení k databázi pomocí SQLAlchemy engine
engine = create_engine('mysql+pymysql://root:@0.tcp.eu.ngrok.io:15383/sarfad')

# Mapování tříd na databázové tabulky
metadata = MetaData()

# Definice databázové tabulky pro ukládání dat o rychlosti
speed_data = Table('SpeedData', metadata,
                   Column('id', Integer, primary_key=True),
                   Column('Time', Float),
                   Column('Speed', Float))

# Vytvoření tabulky v databázi
metadata.create_all(engine)

# Definice třídy pro reprezentaci záznamu o rychlosti
class SpeedData(object):
    def __init__(self, Time, Speed):
        self.Time = Time
        self.Speed = Speed

# Mapování třídy na databázovou tabulku
mapper_registry.map_imperatively(SpeedData, speed_data)

# Vytvoření relační session pro práci s databází
Session = sessionmaker(bind=engine)
session = Session()

# Otevření videa pro analýzu pohybu
cap = cv2.VideoCapture('video.mp4')

# Kontrola, zda se podařilo otevřít video soubor
if not cap.isOpened():
    print("Error: Unable to open video file")
    exit()

# Inicializace proměnných pro sledování pohybu a času
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
distance_traveled = 0
start_time = time.time()
previous_time = start_time
frame_count = 0

# Nastavení intervalu pro výpočet průměrné rychlosti 
speed_interval = 5  

# Nastavení faktoru škálování pro výpočet rychlosti
scale_factor = 10 / 5

# Funkce pro vyhlazení průměrných rychlostí
def smooth_speed(speeds, window_size=5):
    smoothed_speeds = np.convolve(speeds, np.ones(window_size)/window_size, mode='valid')
    return smoothed_speeds

# Pole pro ukládání průměrných rychlostí
speeds = []

# Hlavní smyčka pro analýzu každého snímku videa
try:
    while(cap.isOpened()):
        ret, frame2 = cap.read()
        if not ret:
            break
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Výpočet optického toku mezi aktuálním a předchozím snímkem
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Výpočet magnitudy pohybu
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)

        # Výpočet průměrného pohybu
        avg_motion = np.mean(mag)

        # Aktualizace ujeté vzdálenosti
        distance_traveled += avg_motion

        # Aktualizace předchozího snímku a počtu snímků
        prvs = next_frame
        frame_count += 1

        # Výpočet aktuálního času a uplynulého času
        current_time = time.time()
        elapsed_time = current_time - previous_time

        # Výpočet průměrné rychlosti po uplynutí určeného intervalu
        if elapsed_time >= speed_interval:
            speed_mps = round(distance_traveled / speed_interval / scale_factor, 2)

            # Výpis aktuální rychlosti
            print(f"Instantaneous speed at {round(frame_count / 60, 2)} seconds:", speed_mps, "meters per second")

            # Uložení snímku
            cv2.imwrite(f'images/frame_{str(round(frame_count / 60, 2)).replace(".", ",")}.jpg', frame2)

            # Uložení záznamu rychlosti do databáze
            speed_entry = SpeedData(Time=round(frame_count / 60, 2), Speed=speed_mps)
            session.add(speed_entry)
            session.commit()

            # Přidání rychlosti do pole pro vyhlazení
            speeds.append(speed_mps) 

            # Resetování proměnných pro další výpočet
            distance_traveled = 0
            previous_time = current_time

            # Zvýšení faktoru škálování pro další výpočet
            scale_factor += 0.01

finally:
    # Vyhlazení průměrných rychlostí
    speeds = smooth_speed(speeds)  

    # Uzavření relační session a uvolnění videa
    session.close()
    cap.release()
