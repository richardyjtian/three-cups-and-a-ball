# Packages needed: opencv-python, opencv-contrib-python
import cv2

def drawBBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2, 1)
    cv2.putText(img, "Object Tracking", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

if __name__ == "__main__":
    capture = cv2.VideoCapture(0)
    success, img = capture.read()
    tracker = cv2.TrackerMOSSE_create()
    bbox = cv2.selectROI("Object to Track", img, False)
    tracker.init(img, bbox)

    while True:
        success, img = capture.read()
        success, bbox = tracker.update(img)

        if success:
            # bbox is (x, y, w, h)
            drawBBox(img, bbox)
        else:
            cv2.putText(img, "Object Lost", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show image
        cv2.imshow("Object Tracker", img)

        # Quit when press Q key
        if cv2.waitKey(1) & 0xff == ord('q'):
            break