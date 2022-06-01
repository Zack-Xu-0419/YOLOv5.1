import time

from detectLiteC import det

detector = det(weights="yolov5n.pt")

while True:
    detector.run(10)
    print("Sleeping... Doing something else...")
    time.sleep(3)
