# =============================================
# face_utils.py - Funciones principales
# Detección facial con MediaPipe Tasks API (moderna)
# Compatible con mediapipe >= 0.10.31, Python 3.10+
# =============================================

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# =============================================
# MODELO TFLITE - se descarga automáticamente
# =============================================

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_PATH = "blaze_face_short_range.tflite"


def descargar_modelo_si_falta() -> str:
    """
    Descarga el modelo TFLite de Google si no existe localmente.

    MediaPipe Tasks (API moderna >= 0.10.31) eliminó la Solutions API
    (mp.solutions) y ahora requiere un archivo .tflite explícito.
    El modelo BlazeFace pesa ~300 KB y se descarga solo una vez.
    """
    if not os.path.exists(MODEL_PATH):
        print(f"[INFO] Descargando modelo facial: {MODEL_PATH} ...")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"[OK] Modelo descargado correctamente.")
        except Exception as e:
            raise RuntimeError(
                f"Error al descargar el modelo.\n"
                f"Descárgalo manualmente desde:\n{MODEL_URL}\n"
                f"Y colócalo en la misma carpeta que main.py.\nError: {e}"
            )
    return MODEL_PATH


def crear_detector_imagen(confianza: float = 0.4) -> mp_vision.FaceDetector:
    """
    Crea un detector facial para imágenes estáticas (IMAGE mode).

    Args:
        confianza: Umbral mínimo de confianza (0.0-1.0)
    Returns:
        FaceDetector listo para usar con .detect()
    """
    ruta = descargar_modelo_si_falta()
    base_options = mp_python.BaseOptions(model_asset_path=ruta)
    opciones = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        min_detection_confidence=confianza,
        min_suppression_threshold=0.3
    )
    return mp_vision.FaceDetector.create_from_options(opciones)


def crear_detector_video(confianza: float = 0.4) -> mp_vision.FaceDetector:
    """
    Crea un detector facial para video en tiempo real (VIDEO mode).

    El modo VIDEO usa tracking entre frames para mayor eficiencia.
    Requiere timestamps crecientes en cada llamada.

    Args:
        confianza: Umbral mínimo de confianza (0.0-1.0)
    Returns:
        FaceDetector listo para usar con .detect_for_video()
    """
    ruta = descargar_modelo_si_falta()
    base_options = mp_python.BaseOptions(model_asset_path=ruta)
    opciones = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        min_detection_confidence=confianza,
        min_suppression_threshold=0.3
    )
    return mp_vision.FaceDetector.create_from_options(opciones)


def obtener_coordenadas_cara(detection, h: int, w: int, margen: float = 0.10):
    """
    Convierte el bounding box de MediaPipe Tasks a coordenadas de píxeles.

    En la API moderna (Tasks), las coordenadas ya vienen en píxeles absolutos:
        detection.bounding_box.origin_x  -> x superior izquierdo
        detection.bounding_box.origin_y  -> y superior izquierdo
        detection.bounding_box.width     -> ancho en píxeles
        detection.bounding_box.height    -> alto en píxeles

    Args:
        detection: Objeto Detection de MediaPipe Tasks
        h, w:      Dimensiones de la imagen en píxeles
        margen:    Porcentaje de expansión del bounding box
    Returns:
        (x1, y1, x2, y2) o None si las coordenadas son inválidas
    """
    try:
        bbox = detection.bounding_box
        x1 = int(bbox.origin_x)
        y1 = int(bbox.origin_y)
        x2 = int(bbox.origin_x + bbox.width)
        y2 = int(bbox.origin_y + bbox.height)

        mx = int(bbox.width  * margen)
        my = int(bbox.height * margen)

        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x2 + mx)
        y2 = min(h, y2 + my)

        if x1 >= x2 or y1 >= y2:
            return None
        return (x1, y1, x2, y2)
    except Exception:
        return None


def aplicar_blur_gaussian(imagen: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Aplica GaussianBlur sobre la región del rostro.

    El kernel es dinámico: se ajusta al tamaño del rostro.
    Se aplica 2 veces para un efecto más fuerte y uniforme.
    El kernel siempre debe ser impar (requerimiento de OpenCV).
    """
    resultado = imagen.copy()
    region = resultado[y1:y2, x1:x2]
    if region.size == 0:
        return resultado

    base   = max(y2 - y1, x2 - x1)
    kernel = max(51, int(base * 0.65))
    if kernel % 2 == 0:
        kernel += 1

    blurred = cv2.GaussianBlur(region,   (kernel, kernel), sigmaX=0)
    blurred = cv2.GaussianBlur(blurred,  (kernel, kernel), sigmaX=0)

    resultado[y1:y2, x1:x2] = blurred
    return resultado


def aplicar_pixelado(imagen: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                     intensidad: int = 14) -> np.ndarray:
    """
    Aplica efecto de pixelado (censura) sobre la región del rostro.

    Técnica: reducir resolución y escalar con INTER_NEAREST (sin suavizado).
    """
    resultado = imagen.copy()
    region = resultado[y1:y2, x1:x2]
    if region.size == 0:
        return resultado

    alto  = y2 - y1
    ancho = x2 - x1
    ph = max(1, alto  // intensidad)
    pw = max(1, ancho // intensidad)

    pequeño  = cv2.resize(region,   (pw, ph), interpolation=cv2.INTER_LINEAR)
    pixelado = cv2.resize(pequeño, (ancho, alto), interpolation=cv2.INTER_NEAREST)

    resultado[y1:y2, x1:x2] = pixelado
    return resultado


def procesar_frame_con_detector(frame: np.ndarray, detector,
                                timestamp_ms: int,
                                modo_blur: str = "gaussian") -> tuple:
    """
    Procesa un frame de webcam usando el detector en modo VIDEO.

    Flujo:
    1. BGR (OpenCV) -> RGB -> mp.Image
    2. detector.detect_for_video(mp_img, timestamp_ms)
    3. Por cada cara: extraer bbox -> aplicar efecto -> dibujar overlay

    Args:
        frame:        Frame BGR de OpenCV
        detector:     FaceDetector en modo VIDEO
        timestamp_ms: Timestamp en ms (debe ser creciente entre frames)
        modo_blur:    "gaussian" | "pixelado"
    Returns:
        (frame_procesado, n_caras)
    """
    h, w = frame.shape[:2]
    resultado = frame.copy()

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    detection_result = detector.detect_for_video(mp_img, timestamp_ms)

    n_caras = 0
    if detection_result.detections:
        for det in detection_result.detections:
            coords = obtener_coordenadas_cara(det, h, w)
            if coords is None:
                continue

            x1, y1, x2, y2 = coords
            n_caras += 1

            if modo_blur == "gaussian":
                resultado = aplicar_blur_gaussian(resultado, x1, y1, x2, y2)
            else:
                resultado = aplicar_pixelado(resultado, x1, y1, x2, y2)

            cv2.rectangle(resultado, (x1, y1), (x2, y2), (0, 140, 255), 2)

            score = det.categories[0].score if det.categories else 0.0
            etiqueta = f"CARA {n_caras} [{score:.0%}]"
            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(resultado, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 140, 255), -1)
            cv2.putText(resultado, etiqueta, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return resultado, n_caras


def procesar_imagen_archivo(ruta: str, modo_blur: str = "gaussian",
                            confianza: float = 0.4) -> tuple:
    """
    Carga una imagen, detecta rostros y aplica blur/pixelado.

    Args:
        ruta:      Ruta al archivo de imagen
        modo_blur: "gaussian" | "pixelado"
        confianza: Confianza mínima del detector
    Returns:
        (imagen_original_bgr, imagen_procesada_bgr, n_caras)
        o (None, None, 0) en caso de error
    """
    imagen = cv2.imread(ruta)
    if imagen is None:
        return None, None, 0

    h, w = imagen.shape[:2]
    max_dim = 1280
    if max(h, w) > max_dim:
        escala = max_dim / max(h, w)
        imagen = cv2.resize(imagen, (int(w * escala), int(h * escala)),
                            interpolation=cv2.INTER_AREA)

    h, w = imagen.shape[:2]
    resultado = imagen.copy()

    detector = crear_detector_imagen(confianza)

    frame_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = detector.detect(mp_img)
    detector.close()

    n_caras = 0
    if detection_result.detections:
        for det in detection_result.detections:
            coords = obtener_coordenadas_cara(det, h, w)
            if coords is None:
                continue

            x1, y1, x2, y2 = coords
            n_caras += 1

            if modo_blur == "gaussian":
                resultado = aplicar_blur_gaussian(resultado, x1, y1, x2, y2)
            else:
                resultado = aplicar_pixelado(resultado, x1, y1, x2, y2)

            cv2.rectangle(resultado, (x1, y1), (x2, y2), (0, 140, 255), 2)
            score = det.categories[0].score if det.categories else 0.0
            etiqueta = f"CARA {n_caras} [{score:.0%}]"
            (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(resultado, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 140, 255), -1)
            cv2.putText(resultado, etiqueta, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return imagen, resultado, n_caras


def agregar_overlay_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Agrega contador de FPS con estilo industrial."""
    resultado = frame.copy()
    overlay = resultado.copy()
    cv2.rectangle(overlay, (8, 8), (130, 38), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, resultado, 0.3, 0, resultado)
    cv2.putText(resultado, f"FPS: {fps:.1f}", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2, cv2.LINE_AA)
    return resultado


def agregar_overlay_modo(frame: np.ndarray, modo: str, n_caras: int) -> np.ndarray:
    """Agrega etiqueta del modo activo y número de caras detectadas."""
    resultado = frame.copy()
    h, w = resultado.shape[:2]
    texto = ("BLUR GAUSSIANO" if modo == "gaussian" else "PIXELADO") + f" | Caras: {n_caras}"
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    xi = w - tw - 20
    overlay = resultado.copy()
    cv2.rectangle(overlay, (xi - 5, h - 32), (w - 5, h - 5), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, resultado, 0.3, 0, resultado)
    cv2.putText(resultado, texto, (xi, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    return resultado