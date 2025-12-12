from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
import re

class PlateRecognizer:
    def __init__(self, model_path="best.pt", tesseract_path=None):
        self.model = YOLO(model_path)
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    @staticmethod
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def correct_plate_text(text):
        text = text.upper()

        replacements = {
            'О': 'O', 'А': 'A', 'В': 'B', 'Е': 'E',
            'Н': 'H', 'К': 'K', 'М': 'M', 'Р': 'P',
            'С': 'C', 'Т': 'T', 'Х': 'X',
            'І': 'I', 'Ї': 'I', 'Й': 'I'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)

        text = text.replace('0', 'O')  
        text = text.replace('O', '0', 1) if re.match(r'[A-Z]{2}O\d', text) else text  
        text = re.sub(r'[^A-Z0-9]', '', text)

        match = re.search(r'^[A-Z]{2}\d{4}[A-Z]{2}$', text)
        if match:
            return match.group(0)

        return "Невпізнано"

    def recognize_plate(self, image_path):
        img = cv2.imread(image_path)
        results = self.model(img)
        recognized_texts = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate = img[y1:y2, x1:x2]

            gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                c = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box_pts = cv2.boxPoints(rect).astype(np.float32)
                box_pts_sorted = self.order_points(box_pts)

                width = int(rect[1][0])
                height = int(rect[1][1])
                if width < height:
                    width, height = height, width
                    dst_pts = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(box_pts_sorted[[1,0,3,2]], dst_pts)
                else:
                    dst_pts = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
                    M = cv2.getPerspectiveTransform(box_pts_sorted, dst_pts)

                plate_corrected = cv2.warpPerspective(plate, M, (width, height))
            else:
                plate_corrected = plate

            plate_gray = cv2.cvtColor(plate_corrected, cv2.COLOR_BGR2GRAY)
            plate_resized = cv2.resize(plate_gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            plate_invert = cv2.bitwise_not(plate_resized)

            text_normal = pytesseract.image_to_string(
                plate_invert,
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            ).strip()

            plate_flip = cv2.flip(plate_invert, 1)
            text_flip = pytesseract.image_to_string(
                plate_flip,
                config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            ).strip()

            best_text = text_normal if len(text_normal) >= len(text_flip) else text_flip
            best_text = self.correct_plate_text(best_text)
            recognized_texts.append(best_text)

        if not recognized_texts:
            return "Невпізнано"
        elif len(recognized_texts) == 1:
            return recognized_texts[0]
        else:
            return recognized_texts
        


