import cv2
import numpy as np

img = np.zeros((300, 500, 3), dtype=np.uint8)
cv2.putText(img, "Window OK", (60, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
cv2.imshow("test", img)
print("Press any key in the window to close it")
cv2.waitKey(0)
cv2.destroyAllWindows()