import cv2
print("Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("ERROR: Camera not found!")
    input("Press Enter to close...")
else:
    print("Camera OK! You should see a window now.")
    print("Press Q inside the window to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame!")
            break
        cv2.imshow("CAMERA TEST - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")
