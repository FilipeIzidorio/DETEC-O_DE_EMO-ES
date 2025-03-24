import cv2


class Camera:
    def __init__(self):
        self.captura = cv2.VideoCapture(0)
        if not self.captura.isOpened():
            raise ValueError("Não foi possível abrir a câmera")

    def obter_frame(self):
        ret, frame = self.captura.read()
        if ret:
            return frame
        return None

    def liberar(self):
        if self.captura.isOpened():
            self.captura.release()