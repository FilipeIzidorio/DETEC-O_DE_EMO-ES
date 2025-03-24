from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import os
from deepface import DeepFace
import cv2
from werkzeug.utils import secure_filename
import time
from threading import Thread, Lock

app = Flask(__name__)

# Configurações
PASTA_UPLOAD = 'uploads'
os.makedirs(PASTA_UPLOAD, exist_ok=True)

# Variáveis globais para compartilhamento entre threads
frame_atual = None
lock = Lock()
emocao_atual = "Nenhuma emoção detectada"
camera_ativa = False
camera_thread = None
captura = None


class Camera:
    def __init__(self):
        global captura
        self.captura = cv2.VideoCapture(0)
        if not self.captura.isOpened():
            raise ValueError("Não foi possível abrir a câmera")
        captura = self.captura

    def obter_frame(self):
        ret, frame = self.captura.read()
        if ret:
            return frame
        return None

    def liberar(self):
        if self.captura.isOpened():
            self.captura.release()


@app.route('/')
def pagina_inicial():
    return render_template('index.html')


@app.route('/detectar_imagem', methods=['POST'])
def detectar_imagem():
    if 'imagem' not in request.files:
        return jsonify({'erro': 'Nenhuma imagem enviada'}), 400

    arquivo = request.files['imagem']
    if arquivo.filename == '':
        return jsonify({'erro': 'Nome de arquivo vazio'}), 400

    try:
        # Gera nome seguro para o arquivo
        nome_arquivo = secure_filename(arquivo.filename)
        if not nome_arquivo:
            nome_arquivo = f"imagem_{int(time.time())}.jpg"

        timestamp = int(time.time())
        nome_base = f"{timestamp}_{nome_arquivo}"
        caminho_original = os.path.join(PASTA_UPLOAD, nome_base)

        # Salva a imagem original
        arquivo.save(caminho_original)

        # Processamento da imagem
        img = cv2.imread(caminho_original)
        if img is None:
            return jsonify({'erro': 'Não foi possível ler a imagem'}), 400

        # Análise de emoção
        resultado = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        emocao = resultado[0]['dominant_emotion']

        # Cria versão analisada
        img_analisada = img.copy()
        cv2.putText(img_analisada, f"Emoção: {emocao}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Salva imagem analisada
        nome_analisada = f"analisada_{nome_base}"
        caminho_analisada = os.path.join(PASTA_UPLOAD, nome_analisada)
        cv2.imwrite(caminho_analisada, img_analisada)

        return jsonify({
            'status': 'sucesso',
            'emocao': emocao,
            'imagem_original': nome_base,
            'imagem_analisada': nome_analisada
        })

    except Exception as e:
        return jsonify({'erro': str(e)}), 500


@app.route('/uploads/<nome_arquivo>')
def servir_imagem(nome_arquivo):
    return send_from_directory(PASTA_UPLOAD, nome_arquivo)


def obter_frame():
    global frame_atual, lock
    with lock:
        if frame_atual is None:
            return None
        _, buffer = cv2.imencode('.jpg', frame_atual)
        return buffer.tobytes()


def gerar_frames():
    while camera_ativa:
        frame = obter_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)  # ~30 FPS


@app.route('/video_feed')
def video_feed():
    return Response(gerar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/iniciar_camera', methods=['POST'])
def iniciar_camera():
    global camera_ativa, camera_thread, captura

    if camera_ativa:
        return jsonify({'status': 'já ativa'})

    try:
        # Libera a câmera se já estiver em uso
        if captura is not None:
            captura.release()

        camera_ativa = True
        camera_thread = Thread(target=iniciar_deteccao_camera)
        camera_thread.start()
        return jsonify({'status': 'camera iniciada'})
    except Exception as e:
        return jsonify({'erro': str(e)}), 500


@app.route('/parar_camera', methods=['POST'])
def parar_camera():
    global camera_ativa, camera_thread, captura

    if not camera_ativa:
        return jsonify({'status': 'já parada'})

    camera_ativa = False
    if camera_thread is not None:
        camera_thread.join()
    if captura is not None:
        captura.release()
        captura = None
    return jsonify({'status': 'camera parada'})


@app.route('/obter_emocao', methods=['GET'])
def obter_emocao():
    global emocao_atual
    return jsonify({'emocao': emocao_atual})


def iniciar_deteccao_camera():
    global frame_atual, emocao_atual, camera_ativa, lock, captura

    try:
        camera = Camera()

        while camera_ativa:
            frame = camera.obter_frame()

            if frame is None:
                continue

            try:
                # Redimensiona para melhor performance
                frame_pequeno = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

                # Análise de emoção
                resultado = DeepFace.analyze(frame_pequeno, actions=['emotion'], enforce_detection=False)
                emocao = resultado[0]['dominant_emotion']
                emocao_atual = emocao

                # Desenha resultados no frame
                cv2.putText(frame, f"Emoção: {emocao}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except Exception as e:
                emocao_atual = "Analisando..."

            # Atualiza frame global
            with lock:
                frame_atual = frame

        camera.liberar()
    except Exception as e:
        print(f"Erro na thread da câmera: {str(e)}")
        camera_ativa = False


if __name__ == '__main__':
    app.run(debug=True, threaded=True)