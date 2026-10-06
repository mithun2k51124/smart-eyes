import cv2

print("Scanning for cameras...")
found = []
for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW = Windows DirectShow
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"  [FOUND] Camera index {i}  ->  {w}x{h}")
            found.append(i)
        else:
            print(f"  [OPEN but no frame] index {i}")
        cap.release()
    else:
        print(f"  [NOT found] index {i}")

if not found:
    print("\nNo cameras detected. Check USB connection and Windows camera privacy settings.")
else:
    print(f"\nBest camera index to use: {found[0]}")
    print(f"-> Set SOURCE = {found[0]} in fish_config.py")
