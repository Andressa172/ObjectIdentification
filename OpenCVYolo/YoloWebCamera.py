import numpy as np
import cv2
import time
import pyttsx3
import threading
from csv import DictWriter 
import subprocess

# Inicializar o pyttsx3 com o driver 'espeak' para garantir compatibilidade no Linux
engine = pyttsx3.init(driverName='espeak')

# Definir a configuração da fala
engine.setProperty('rate', 250)  # Ajuste a velocidade da fala
engine.setProperty('volume', 1)  # Ajuste o volume (0.0 a 1.0)

# Função para falar o nome do objeto
def speak(text):
    subprocess.run(["espeak", "-s", "150", "-v", "pt+f5", text])  # Velocidade reduzida e voz feminina

# Definir qual câmera será utilizada na captura
camera = cv2.VideoCapture(0)

# Cria variáveis para captura de altura e largura
h, w = None, None

# Carrega o arquivo com os nomes dos objetos que o arquivo foi treinado para detectar
with open('yoloDados/YoloNames.names') as f:
    labels = [line.strip() for line in f]

# Carrega os arquivos treinados pelo framework
network = cv2.dnn.readNetFromDarknet('yoloDados/yolov3.cfg', 'yoloDados/yolov3.weights')

# Captura uma lista com todos os nomes de objetos treinados pelo framework
layers_names_all = network.getLayerNames()
output_layers = network.getUnconnectedOutLayers().flatten()  # Garante que é um array 1D
output_layers_adjusted = output_layers - 1
layers_names_output = [layers_names_all[int(i)] for i in output_layers_adjusted]

# Definir probabilidade mínima para eliminar previsões fracas
probability_minimum = 0.5
threshold = 0.3  # Limite para filtrar caixas delimitadoras fracas com supressão não máxima

# Gera cores aleatórias nas caixas de cada objeto detectado
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# Variável para controlar a frequência das falas
last_spoken = time.time()

# Loop de captura e detecção dos objetos
with open('teste.csv', 'w') as arquivo:
    cabecalho = ['Detectado', 'Acuracia']
    escritor_csv = DictWriter(arquivo, fieldnames=cabecalho)
    escritor_csv.writeheader()

    while True:
        _, frame = camera.read()

        if w is None or h is None:
            h, w = frame.shape[:2]

        # Processamento de imagem
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        network.setInput(blob)
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()

        # Listas para detectar objetos
        bounding_boxes = []
        confidences = []
        class_numbers = []

        for result in output_from_network:
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]

                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

        if len(results) > 0:
            for i in results.flatten():
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                colour_box_current = colours[class_numbers[i]].tolist()
                cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 2)

                # Preparando texto com rótulo e acurácia para o objeto detectado
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])
                cv2.putText(frame, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

                escritor_csv.writerow({"Detectado": text_box_current.split(':')[0], "Acuracia": text_box_current.split(':')[1]})
                print(text_box_current.split(':')[0] + " - " + text_box_current.split(':')[1])

                # Gerar o som do nome do objeto detectado, mas com um delay para evitar fala repetida
                if time.time() - last_spoken > 1:  # Falar a cada 1 segundo (ajuste conforme necessário)
                    threading.Thread(target=speak, args=(labels[int(class_numbers[i])],)).start()  # Falar em thread separada
                    last_spoken = time.time()

        cv2.namedWindow('YOLO v3 WebCamera', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLO v3 WebCamera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

camera.release()
cv2.destroyAllWindows()
