import cv2
import numpy as np
from ultralytics import YOLO
import math

FOV_DEG = 110.0
WIDTH   = 1280
FX = (WIDTH / 2) / math.tan(math.radians(FOV_DEG / 2))  # 448.14
CX = WIDTH / 2   # 640.0
CY = 360.0

CONE_HEIGHT_M  = 0.325  # altura real do cone FSDS em metros
CAMERA_HEIGHT_M = 1.0   # altura da câmera no settings.json (Z=1.0)

CONF_THRESHOLD = 0.6
MODEL_PATH = "/home/demarco/16_01/best.pt"

DEBUG = False

class Perception:
    def __init__(self, client):
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"YOLO failed to start.\n{e}")
        self.client = client

    def estimateDistance(self, box_h_pixels):
        """Estima distância pelo tamanho vertical do bounding box."""
        if box_h_pixels < 1:
            return None
        dist = (CONE_HEIGHT_M * FX) / box_h_pixels
        return dist

    def detectCones(self, img):
        image = cv2.imdecode(
            np.frombuffer(img.image_data_uint8, dtype=np.uint8), cv2.IMREAD_COLOR)

        if image is None:
            return np.array([])

        results = self.model(image, verbose=False, conf=CONF_THRESHOLD)
        boxes = results[0].boxes

        if DEBUG:
            print(f"\nYOLO: {len(boxes)} detecções")

        rawDetection = []

        for box in boxes:
            u    = float(box.xywh[0][0])   # centro x
            h_px = float(box.xywh[0][3])   # altura do bbox em pixels
            classId = int(box.cls[0])
            conf    = float(box.conf[0])

            dist = self.estimateDistance(h_px)
            if dist is None:
                continue

            # distância é x_frente (frente do carro)
            x_frente = dist
            # lateral pelo ângulo horizontal
            y_lateral = (u - CX) * x_frente / FX

            if DEBUG:
                print(f"  cls={classId} conf={conf:.2f} u={u:.0f} "
                      f"h={h_px:.0f}px → x={x_frente:.1f}m y={y_lateral:.1f}m")

            if 0.5 < x_frente < 35.0:
                rawDetection.append([y_lateral, x_frente, classId])

        finalCones = []
        for cone in rawDetection:
            if not any(np.linalg.norm(np.array(cone[:2]) - np.array(f[:2])) < 1.2
                       for f in finalCones):
                finalCones.append(cone)

        if DEBUG:
            print(f"Cones finais: {len(finalCones)}")

        return np.array(finalCones)  # [lateral, profundidade, id]