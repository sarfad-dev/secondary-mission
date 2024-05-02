import cv2
import numpy as np
import time
import os
from sqlalchemy import create_engine, Column, Float, Integer, MetaData, Table
from sqlalchemy.orm import registry, sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Inicializace registru mapperu pro SQLAlchemy, který umožňuje mapování tříd na databázové tabulky
mapper_registry = registry()

db_username = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

db_url = f'mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}'

engine = create_engine(db_url)

# Inicializace metadat pro mapování tříd na databázové tabulky
metadata = MetaData()

# Definice databázové tabulky pro ukládání dat o rychlosti. Obsahuje sloupce: id (primární klíč), Time a Speed.
speed_data = Table('SpeedData', metadata,
                   Column('id', Integer, primary_key=True),
                   Column('Time', Float),
                   Column('Speed', Float))

# Vytvoření tabulky v databázi pomocí metadat a engine
metadata.create_all(engine)

# Definice třídy SpeedData pro reprezentaci záznamu o rychlosti
class SpeedData(object):
    def __init__(self, Time, Speed):
        self.Time = Time
        self.Speed = Speed

# Namapování třídy SpeedData na databázovou tabulku SpeedData
mapper_registry.map_imperatively(SpeedData, speed_data)

# Vytvoření sessionmakeru, který umožňuje vytvoření relační session pro práci s databází
Session = sessionmaker(bind=engine)
session = Session()

# Otevření videa pro analýzu pohybu pomocí OpenCV
cap = cv2.VideoCapture('video.mp4')

# Kontrola, jestli se podařilo otevřít video soubor
if not cap.isOpened():
    print("Error: Unable to open video file")
    exit()

# Inicializace proměnných pro sledování pohybu a času
# Přečtení prvního snímku videa a jeho konverze do šedé škály, z důvodu lehčí analýzy pomocí OpenCV
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Inicializace proměnných pro ujetou vzdálenost, začátek času, předchozí čas a počet snímků
distance_traveled = 0
start_time = time.time()
previous_time = start_time
frame_count = 0

# Nastavení intervalu pro výpočet průměrné rychlosti v sekundách
speed_interval = 5  

# Nastavení faktoru škálování pro výpočet rychlosti
scale_factor = 10 / 5

# Funkce pro vyhlazení průměrných rychlostí
def smooth_speed(speeds, window_size=5):
    # Používá konvoluční operaci pro vyhlazení pole rychlostí
    smoothed_speeds = np.convolve(speeds, np.ones(window_size)/window_size, mode='valid')
    return smoothed_speeds

# Pole pro ukládání průměrných rychlostí
speeds = []

# Hlavní smyčka pro analýzu každého snímku videa
try:
    while cap.isOpened():
        # Přečtení dalšího snímku z videa
        ret, frame2 = cap.read()
        if not ret:
            break
        
        # Konverze aktuálního snímku do šedé škály
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Výpočet optického toku mezi aktuálním a předchozím snímkem pomocí algoritmu Farneback
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Výpočet magnitudy pohybu pomocí výpočtu euklidovské vzdálenosti mezi složkami optického toku
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # Výpočet průměrného pohybu, což je průměrná magnituda toku ve snímku
        avg_motion = np.mean(mag)

        # Aktualizace ujeté vzdálenosti na základě průměrného pohybu
        distance_traveled += avg_motion

        # Aktualizace předchozího snímku pro další iteraci smyčky
        prvs = next_frame
        frame_count += 1

        # Výpočet aktuálního času a uplynulého času od posledního měření
        current_time = time.time()
        elapsed_time = current_time - previous_time

        # Pokud uplynul určený interval, vypočítáme průměrnou rychlost za tento interval
        if elapsed_time >= speed_interval:
            # Výpočet průměrné rychlosti v metrech za sekundu
            speed_mps = round(distance_traveled / speed_interval / scale_factor, 2)

            print(f"Instantaneous speed at {round(frame_count / 60, 2)} seconds:", speed_mps, "meters per second")

            # Uložení jednotlivého snímku
            cv2.imwrite(f'images/frame_{str(round(frame_count / 60, 2)).replace(".", ",")}.jpg', frame2)

            # Uložení záznamu rychlosti do databáze
            speed_entry = SpeedData(Time=round(frame_count / 60, 2), Speed=speed_mps)
            session.add(speed_entry)
            session.commit()

            speeds.append(speed_mps)

            # Resetování ujeté vzdálenosti pro další loop
            distance_traveled = 0
            
            # Aktualizace předchozího času pro další interval
            previous_time = current_time

            # Zvyšování faktoru škálování o malou hodnotu v každém intervalu pro přesnější výpočet rychlosti - cansat padá = scale factor se musí měnit
            scale_factor += 0.01

finally:
    speeds = smooth_speed(speeds)

    session.close()
    cap.release()
