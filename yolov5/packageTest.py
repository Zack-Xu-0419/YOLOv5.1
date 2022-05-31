from time import sleep, time
from detectLiteC import det
import time

detector = det(weights="yolov5n.pt")

while(True):
    detector.run(10)
    print("Sleeping... Doing something else...")
    time.sleep(3)
