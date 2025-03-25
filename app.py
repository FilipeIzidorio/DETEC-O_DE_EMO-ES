from flask import Flask, render_template, request, jsonify, Response, send_from_directory
import os
from deepface import DeepFace
import cv2
from werkzeug.utils import secure_filename
import time
from threading import Thread, Lock
import numpy as np

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

def analisar_emocao_rosto(rosto, x, y, w, h, resultados, marcacoes, index, lock):
    try:
        print(f"[Thread {index}] Analisando rosto na posição ({x}, {y}, {w}, {h})")
        resultado = DeepFace.analyze(rosto, actions=['emotion'], enforce_detection=False)

        if isinstance(resultado, list) and len(resultado) > 0 and 'dominant_emotion' in resultado[0]:
            emocao = resultado[0]['dominant_emotion']
        else:
            emocao = "Desconhecida"
    except Exception as e:
        print(f"[Thread {index}] Erro ao analisar emoção: {e}")
        emocao = "Erro"

    with lock:
        resultados.append({
            'posicao': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
            'emocao': emocao
        })
        marcacoes.append({
            'x': int(x),
            'y': int(y),
            'w': int(w),
            'h': int(h),
            'emocao': emocao
        })

def detectar_rostos_e_emocoes(img):
    resultados = []
    marcacoes = []
    threads = []
    lock = Lock()

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    img_cinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cinza = cv2.equalizeHist(img_cinza)

    rostos = face_cascade.detectMultiScale(
        img_cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(f"[INFO] {len(rostos)} rosto(s) detectado(s)")

    for i, (x, y, w, h) in enumerate(rostos):
        rosto = img[y:y + h, x:x + w]
        t = Thread(target=analisar_emocao_rosto, args=(rosto, x, y, w, h, resultados, marcacoes, i, lock))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    for m in marcacoes:
        x, y, w, h = m['x'], m['y'], m['w'], m['h']
        # Alterar a cor do retângulo (BGR: azul, verde, vermelho)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde
        # Alterar a cor do texto (BGR: azul, verde, vermelho)
        cv2.putText(img, m['emocao'], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Vermelho

    # 🔁 Conversão para JSON
    for r in resultados:
        r['posicao'] = {k: int(v) for k, v in r['posicao'].items()}
    for m in marcacoes:
        for k in ['x', 'y', 'w', 'h']:
            m[k] = int(m[k])

    return resultados, img, marcacoes

@app.route('/detectar_imagem', methods=['POST'])
def detectar_imagem():
    if 'imagem' not in request.files:
        return jsonify({'erro': 'Nenhuma imagem enviada'}), 400

    arquivo = request.files['imagem']
    if arquivo.filename == '':
        return jsonify({'erro': 'Nome de arquivo vazio'}), 400

    try:
        # Gera nome seguro
        nome_arquivo = secure_filename(arquivo.filename)
        if not nome_arquivo:
            nome_arquivo = f"imagem_{int(time.time())}.jpg"

        timestamp = int(time.time())
        nome_base = f"{timestamp}_{nome_arquivo}"
        caminho_original = os.path.join(PASTA_UPLOAD, nome_base)

        # Salva a imagem original
        arquivo.save(caminho_original)
        print(f"[INFO] Imagem salva: {caminho_original}")

        # Lê imagem com OpenCV
        img = cv2.imread(caminho_original)
        print(f"[INFO] Tipo da imagem carregada: {type(img)}")
        if img is None:
            print("[ERRO] cv2.imread retornou None.")
            return jsonify({'erro': 'Não foi possível ler a imagem'}), 400

        # Detectar rostos e emoções com threads
        resultados, img_analisada, marcacoes = detectar_rostos_e_emocoes(img)
        print(f"[INFO] Emoções detectadas: {resultados}")

        # Salva imagem com marcações
        nome_analisada = f"analisada_{nome_base}"
        caminho_analisada = os.path.join(PASTA_UPLOAD, nome_analisada)
        cv2.imwrite(caminho_analisada, img_analisada)
        print(f"[INFO] Imagem analisada salva: {caminho_analisada}")

        return jsonify({
            'imagem_analisada': nome_analisada,
            'emocoes': resultados
        }), 200

    except Exception as e:
        print(f"[ERRO] Exceção ao processar imagem: {str(e)}")
        return jsonify({'erro': f'Ocorreu um erro interno: {str(e)}'}), 500

@app.route('/uploads/<nome_arquivo>')
def servir_imagem(nome_arquivo):
    try:
        return send_from_directory(PASTA_UPLOAD, nome_arquivo)
    except Exception as e:
        # Retornar erro como JSON válido
        return jsonify({'erro': f'Ocorreu um erro ao servir a imagem: {str(e)}'}), 500

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
    try:
        return Response(gerar_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        # Retornar erro como JSON válido
        return jsonify({'erro': f'Ocorreu um erro no feed de vídeo: {str(e)}'}), 500

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
        # Retornar erro como JSON válido
        return jsonify({'erro': f'Ocorreu um erro: {str(e)}'}), 500

@app.route('/parar_camera', methods=['POST'])
def parar_camera():
    global camera_ativa, camera_thread, captura

    if not camera_ativa:
        return jsonify({'status': 'já parada'})

    try:
        camera_ativa = False
        if camera_thread is not None:
            camera_thread.join()
        if captura is not None:
            captura.release()
            captura = None
        return jsonify({'status': 'camera parada'})
    except Exception as e:
        # Retornar erro como JSON válido
        return jsonify({'erro': f'Ocorreu um erro: {str(e)}'}), 500

@app.route('/obter_emocao', methods=['GET'])
def obter_emocao():
    global emocao_atual
    return jsonify({'emocao': emocao_atual})

def iniciar_deteccao_camera():
    global frame_atual, emocao_atual, camera_ativa, lock, captura

    try:
        camera = Camera()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        while camera_ativa:
            frame = camera.obter_frame()
            if frame is None:
                continue

            try:
                # Redimensiona para melhorar desempenho
                frame_red = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                img_cinza = cv2.cvtColor(frame_red, cv2.COLOR_BGR2GRAY)
                img_cinza = cv2.equalizeHist(img_cinza)

                rostos = face_cascade.detectMultiScale(
                    img_cinza,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )

                threads = []
                resultados = []
                lock_resultados = Lock()

                for i, (x, y, w, h) in enumerate(rostos):
                    rosto = frame_red[y:y + h, x:x + w]
                    t = Thread(target=analisar_emocao_rosto,
                            args=(rosto, x, y, w, h, resultados, [], i, lock_resultados))
                    t.start()
                    threads.append(t)

                for t in threads:
                    t.join()

                # Atualizar a emoção dominante (primeiro rosto, se houver)
                if resultados:
                    emocao_atual = resultados[0]['emocao']
                else:
                    emocao_atual = "Nenhuma emoção detectada"

                # Desenhar marcações no frame_red
                for res in resultados:
                    pos = res['posicao']
                    x, y, w, h = pos['x'], pos['y'], pos['w'], pos['h']
                    emocao = res['emocao']
                    # Alterar a cor do retângulo (BGR: azul, verde, vermelho)
                    cv2.rectangle(frame_red, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Verde
                    # Alterar a cor do texto (BGR: azul, verde, vermelho)
                    cv2.putText(frame_red, emocao, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Vermelho

                # Redimensionar de volta ao original
                frame_result = cv2.resize(frame_red, (frame.shape[1], frame.shape[0]))

            except Exception as e:
                print(f"[ERRO] Análise na câmera: {e}")
                frame_result = frame
                emocao_atual = "Analisando..."

            with lock:
                frame_atual = frame_result

        camera.liberar()
    except Exception as e:
        print(f"[ERRO] na thread da câmera: {str(e)}")
        camera_ativa = False

if __name__ == '__main__':
    app.run(debug=True, threaded=True)