from time import sleep, time
from detectLiteC import det
import time

detector = det(weights="yolov5n.pt")

while(True):
    print("Sleeping... Doing something else...")
    time.sleep(5)
    detector.run(100)
