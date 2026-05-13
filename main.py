# =============================================
# main.py - Aplicación principal
# Face Blur Detection - Interfaz Industrial
# Compatible con mediapipe >= 0.10.31, Python 3.10+
# =============================================

import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from face_utils import (
    descargar_modelo_si_falta,
    crear_detector_video,
    procesar_frame_con_detector,
    procesar_imagen_archivo,
    agregar_overlay_fps,
    agregar_overlay_modo,
)

# =============================================
# CONFIGURACIÓN GLOBAL
# =============================================

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# =============================================
# PALETA INDUSTRIAL
# =============================================
COLOR_BG_PRIMARIO   = "#0D0D0D"
COLOR_BG_SECUNDARIO = "#1A1A1A"
COLOR_BG_PANEL      = "#141414"
COLOR_NARANJA       = "#FF6B00"
COLOR_NARANJA_HOVER = "#E05A00"
COLOR_AMARILLO      = "#FFB800"
COLOR_TEXTO         = "#E8E8E8"
COLOR_TEXTO_GRIS    = "#888888"
COLOR_BORDE         = "#2A2A2A"
COLOR_ACENTO        = "#00B4D8"
COLOR_EXITO         = "#2ECC71"
COLOR_ERROR         = "#E74C3C"


# =============================================
# APLICACIÓN PRINCIPAL
# =============================================

class FaceBlurApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("FACE BLUR  |  Sistema de Privacidad Facial")
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.configure(fg_color=COLOR_BG_PRIMARIO)

        # Estado
        self.modo_blur        = "gaussian"
        self.webcam_activa    = False
        self.cap              = None
        self.detector_webcam  = None
        self.thread_webcam    = None
        self.fps_contador     = 0
        self.fps_timer        = time.time()
        self.fps_actual       = 0.0
        self.ts_inicio        = int(time.time() * 1000)   # base para timestamps

        self.imagen_procesada   = None
        self.ruta_imagen_actual = None
        self._photo_actual      = None
        self._pil_frame_actual  = None
        self._n_caras_actual    = 0

        # Construir UI
        self._construir_ui()

        # Pre-descargar modelo al inicio (en background)
        threading.Thread(target=self._precargar_modelo, daemon=True).start()

    # ------------------------------------------
    # PRE-CARGA DEL MODELO
    # ------------------------------------------

    def _precargar_modelo(self):
        """Descarga el modelo en background para que esté listo al usar."""
        try:
            descargar_modelo_si_falta()
            self.after(0, lambda: self._set_status("Modelo cargado — Listo para usar", COLOR_EXITO))
        except Exception as e:
            self.after(0, lambda: self._set_status(f"Error al cargar modelo: {e}", COLOR_ERROR))

    # ------------------------------------------
    # CONSTRUCCIÓN DE UI
    # ------------------------------------------

    def _construir_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._construir_panel_lateral()
        self._construir_panel_display()

    def _construir_panel_lateral(self):
        panel = ctk.CTkFrame(self, width=240, fg_color=COLOR_BG_SECUNDARIO, corner_radius=0)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.grid_propagate(False)
        panel.grid_columnconfigure(0, weight=1)

        # Logo
        frame_logo = ctk.CTkFrame(panel, fg_color=COLOR_BG_PANEL, corner_radius=0, height=90)
        frame_logo.grid(row=0, column=0, sticky="ew")
        frame_logo.grid_propagate(False)
        frame_logo.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame_logo, text="◈ FACE BLUR",
                     font=ctk.CTkFont(family="Courier New", size=20, weight="bold"),
                     text_color=COLOR_NARANJA).grid(row=0, column=0, pady=(16, 0))
        ctk.CTkLabel(frame_logo, text="SISTEMA DE PRIVACIDAD FACIAL",
                     font=ctk.CTkFont(family="Courier New", size=8),
                     text_color=COLOR_TEXTO_GRIS).grid(row=1, column=0, pady=(2, 14))

        ctk.CTkFrame(panel, height=2, fg_color=COLOR_NARANJA, corner_radius=0).grid(
            row=1, column=0, sticky="ew")

        # Sección: Modo entrada
        self._seccion(panel, row=2, texto="■ MODO DE ENTRADA")

        self.btn_webcam = ctk.CTkButton(
            panel, text="▶  ABRIR WEBCAM",
            font=ctk.CTkFont(family="Courier New", size=13, weight="bold"),
            fg_color=COLOR_NARANJA, hover_color=COLOR_NARANJA_HOVER,
            text_color="white", height=44, corner_radius=4,
            command=self._toggle_webcam)
        self.btn_webcam.grid(row=3, column=0, padx=16, pady=(6, 4), sticky="ew")

        self.btn_imagen = ctk.CTkButton(
            panel, text="◉  CARGAR IMAGEN",
            font=ctk.CTkFont(family="Courier New", size=13, weight="bold"),
            fg_color="#1E1E1E", hover_color="#2A2A2A",
            text_color=COLOR_TEXTO, border_color=COLOR_NARANJA, border_width=1,
            height=44, corner_radius=4,
            command=self._cargar_imagen)
        self.btn_imagen.grid(row=4, column=0, padx=16, pady=(4, 6), sticky="ew")

        ctk.CTkFrame(panel, height=1, fg_color=COLOR_BORDE, corner_radius=0).grid(
            row=5, column=0, sticky="ew", padx=16, pady=8)

        # Sección: Efecto
        self._seccion(panel, row=6, texto="■ EFECTO FACIAL")

        self.selector_efecto = ctk.CTkSegmentedButton(
            panel, values=["GAUSSIANO", "PIXELADO"],
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            fg_color=COLOR_BG_PANEL,
            selected_color=COLOR_NARANJA, selected_hover_color=COLOR_NARANJA_HOVER,
            unselected_color=COLOR_BG_PANEL, unselected_hover_color="#1E1E1E",
            text_color=COLOR_TEXTO,
            command=self._cambiar_efecto)
        self.selector_efecto.set("GAUSSIANO")
        self.selector_efecto.grid(row=7, column=0, padx=16, pady=(6, 8), sticky="ew")

        ctk.CTkFrame(panel, height=1, fg_color=COLOR_BORDE, corner_radius=0).grid(
            row=8, column=0, sticky="ew", padx=16, pady=8)

        # Sección: Sensibilidad
        self._seccion(panel, row=9, texto="■ SENSIBILIDAD")

        self.label_confianza = ctk.CTkLabel(
            panel, text="Confianza mínima: 40%",
            font=ctk.CTkFont(family="Courier New", size=10),
            text_color=COLOR_TEXTO_GRIS)
        self.label_confianza.grid(row=10, column=0, padx=16, pady=(4, 2), sticky="w")

        self.slider_confianza = ctk.CTkSlider(
            panel, from_=20, to=90, number_of_steps=70,
            progress_color=COLOR_NARANJA, button_color=COLOR_NARANJA,
            button_hover_color=COLOR_NARANJA_HOVER,
            command=self._actualizar_confianza)
        self.slider_confianza.set(40)
        self.slider_confianza.grid(row=11, column=0, padx=16, pady=(0, 8), sticky="ew")

        ctk.CTkFrame(panel, height=1, fg_color=COLOR_BORDE, corner_radius=0).grid(
            row=12, column=0, sticky="ew", padx=16, pady=8)

        # Sección: Stats
        self._seccion(panel, row=13, texto="■ INFORMACIÓN")

        frame_stats = ctk.CTkFrame(panel, fg_color=COLOR_BG_PANEL, corner_radius=4)
        frame_stats.grid(row=14, column=0, padx=16, pady=(6, 8), sticky="ew")
        frame_stats.grid_columnconfigure(0, weight=1)

        self.lbl_estado = self._stat_label(frame_stats, 0, "ESTADO:", "INACTIVO", COLOR_TEXTO_GRIS)
        self.lbl_caras  = self._stat_label(frame_stats, 1, "CARAS:",  "0",        COLOR_TEXTO)
        self.lbl_fps    = self._stat_label(frame_stats, 2, "FPS:",    "—",        COLOR_ACENTO)
        self.lbl_efecto = self._stat_label(frame_stats, 3, "EFECTO:", "GAUSSIANO",COLOR_NARANJA)

        ctk.CTkFrame(panel, height=1, fg_color=COLOR_BORDE, corner_radius=0).grid(
            row=15, column=0, sticky="ew", padx=16, pady=8)

        self.btn_guardar = ctk.CTkButton(
            panel, text="💾  GUARDAR RESULTADO",
            font=ctk.CTkFont(family="Courier New", size=11, weight="bold"),
            fg_color="#1A2A1A", hover_color="#1E3A1E",
            text_color=COLOR_EXITO, border_color=COLOR_EXITO, border_width=1,
            height=38, corner_radius=4, state="disabled",
            command=self._guardar_imagen)
        self.btn_guardar.grid(row=16, column=0, padx=16, pady=(4, 8), sticky="ew")

        ctk.CTkLabel(panel, text="MediaPipe Tasks + OpenCV + Python",
                     font=ctk.CTkFont(family="Courier New", size=8),
                     text_color="#444444").grid(row=17, column=0, pady=(8, 0))
        ctk.CTkLabel(panel, text="v2.0 — 2025",
                     font=ctk.CTkFont(family="Courier New", size=8),
                     text_color="#444444").grid(row=18, column=0, pady=(0, 12))

    def _construir_panel_display(self):
        panel = ctk.CTkFrame(self, fg_color=COLOR_BG_PRIMARIO, corner_radius=0)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        # Barra superior
        barra = ctk.CTkFrame(panel, fg_color=COLOR_BG_PANEL, height=42, corner_radius=0)
        barra.grid(row=0, column=0, sticky="ew")
        barra.grid_columnconfigure(1, weight=1)
        barra.grid_propagate(False)

        ctk.CTkLabel(barra, text="  ◈  PANEL DE VISUALIZACIÓN",
                     font=ctk.CTkFont(family="Courier New", size=11),
                     text_color=COLOR_TEXTO_GRIS).grid(row=0, column=0, padx=10, sticky="w")

        self.lbl_barra_info = ctk.CTkLabel(barra, text="Sin fuente activa  ",
                                           font=ctk.CTkFont(family="Courier New", size=10),
                                           text_color=COLOR_TEXTO_GRIS)
        self.lbl_barra_info.grid(row=0, column=2, padx=10, sticky="e")

        # Canvas
        self.canvas = tk.Canvas(panel, bg=COLOR_BG_PRIMARIO,
                                highlightthickness=0, cursor="crosshair")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.canvas.bind("<Configure>", self._redimensionar_canvas)
        self._mostrar_pantalla_inicio()

        # Barra inferior
        barra_inf = ctk.CTkFrame(panel, fg_color=COLOR_BG_PANEL, height=32, corner_radius=0)
        barra_inf.grid(row=2, column=0, sticky="ew")
        barra_inf.grid_propagate(False)
        barra_inf.grid_columnconfigure(1, weight=1)

        self.lbl_status_bar = ctk.CTkLabel(
            barra_inf, text="  ● Cargando modelo...",
            font=ctk.CTkFont(family="Courier New", size=9),
            text_color=COLOR_TEXTO_GRIS)
        self.lbl_status_bar.grid(row=0, column=0, padx=6, sticky="w")

        ctk.CTkLabel(barra_inf, text="[Q] = Salir webcam  ",
                     font=ctk.CTkFont(family="Courier New", size=9),
                     text_color="#444444").grid(row=0, column=2, padx=6, sticky="e")

    # ------------------------------------------
    # HELPERS UI
    # ------------------------------------------

    def _seccion(self, parent, row, texto):
        ctk.CTkLabel(parent, text=texto,
                     font=ctk.CTkFont(family="Courier New", size=9, weight="bold"),
                     text_color=COLOR_NARANJA
                     ).grid(row=row, column=0, padx=16, pady=(8, 2), sticky="w")

    def _stat_label(self, parent, row, key, value, color):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row, column=0, padx=8, pady=2, sticky="ew")
        frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frame, text=key,
                     font=ctk.CTkFont(family="Courier New", size=9),
                     text_color=COLOR_TEXTO_GRIS, width=60, anchor="w"
                     ).grid(row=0, column=0, sticky="w")
        lbl = ctk.CTkLabel(frame, text=value,
                           font=ctk.CTkFont(family="Courier New", size=10, weight="bold"),
                           text_color=color, anchor="e")
        lbl.grid(row=0, column=1, sticky="e")
        return lbl

    def _set_status(self, texto, color=None):
        self.lbl_status_bar.configure(
            text=f"  ● {texto}", text_color=color or COLOR_TEXTO_GRIS)

    def _actualizar_stats(self, estado=None, caras=None, fps=None, efecto=None):
        if estado  is not None: self.lbl_estado.configure(text=estado)
        if caras   is not None:
            c = COLOR_EXITO if caras > 0 else COLOR_TEXTO
            self.lbl_caras.configure(text=str(caras), text_color=c)
        if fps     is not None: self.lbl_fps.configure(text=f"{fps:.1f}")
        if efecto  is not None: self.lbl_efecto.configure(text=efecto)

    def _mostrar_pantalla_inicio(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()  or 800
        ch = self.canvas.winfo_height() or 500
        cx, cy = cw // 2, ch // 2
        self.canvas.create_text(cx, cy - 60, text="◈",
                                font=("Courier New", 64), fill="#1E1E1E")
        self.canvas.create_text(cx, cy + 10, text="FACE BLUR DETECTION",
                                font=("Courier New", 18, "bold"), fill="#2A2A2A")
        self.canvas.create_text(cx, cy + 40,
                                text="Selecciona WEBCAM o IMAGEN para comenzar",
                                font=("Courier New", 11), fill="#333333")

    def _redimensionar_canvas(self, event):
        if self.imagen_procesada and not self.webcam_activa:
            self._mostrar_imagen_en_canvas(self.imagen_procesada)

    # ------------------------------------------
    # MODO WEBCAM
    # ------------------------------------------

    def _toggle_webcam(self):
        if self.webcam_activa:
            self._detener_webcam()
        else:
            self._iniciar_webcam()

    def _iniciar_webcam(self):
        self.imagen_procesada = None
        self.btn_guardar.configure(state="disabled")

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error de Cámara",
                                 "No se pudo abrir la webcam.\n"
                                 "Verifica que esté conectada y no esté en uso.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self.cap.set(cv2.CAP_PROP_FPS,            30)

        # Crear detector en modo VIDEO
        confianza = self.slider_confianza.get() / 100.0
        try:
            self.detector_webcam = crear_detector_video(confianza)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo crear el detector:\n{e}")
            self.cap.release()
            return

        self.webcam_activa  = True
        self.fps_contador   = 0
        self.fps_timer      = time.time()
        self.ts_inicio      = int(time.time() * 1000)

        self.btn_webcam.configure(text="■  DETENER WEBCAM",
                                  fg_color=COLOR_ERROR, hover_color="#C0392B")
        self.btn_imagen.configure(state="disabled")
        self._set_status("Webcam activa — Detectando rostros en tiempo real", COLOR_EXITO)
        self._actualizar_stats(estado="ACTIVO", caras=0, fps=0.0)
        self.lbl_barra_info.configure(text="WEBCAM — Tiempo real  ")

        self.thread_webcam = threading.Thread(target=self._loop_webcam, daemon=True)
        self.thread_webcam.start()

    def _loop_webcam(self):
        """
        Loop de captura en thread separado.

        Usa detect_for_video() con timestamps en ms.
        El timestamp debe ser SIEMPRE creciente entre frames.
        """
        while self.webcam_activa and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            # Timestamp en milisegundos desde el inicio de la sesión
            timestamp_ms = int(time.time() * 1000) - self.ts_inicio

            try:
                frame_proc, n_caras = procesar_frame_con_detector(
                    frame, self.detector_webcam, timestamp_ms, self.modo_blur)
            except Exception:
                frame_proc = frame.copy()
                n_caras = 0

            # Calcular FPS
            self.fps_contador += 1
            delta = time.time() - self.fps_timer
            if delta >= 1.0:
                self.fps_actual   = self.fps_contador / delta
                self.fps_contador = 0
                self.fps_timer    = time.time()

            # Overlays
            frame_proc = agregar_overlay_fps(frame_proc, self.fps_actual)
            frame_proc = agregar_overlay_modo(frame_proc, self.modo_blur, n_caras)

            # BGR -> RGB -> PIL
            frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            self._pil_frame_actual = Image.fromarray(frame_rgb)
            self._n_caras_actual   = n_caras

            self.after(0, self._actualizar_frame_webcam)
            time.sleep(0.001)

        if self.webcam_activa:
            self.after(0, self._detener_webcam)

    def _actualizar_frame_webcam(self):
        if not self.webcam_activa:
            return
        try:
            pil = self._pil_frame_actual
            if pil:
                self._mostrar_imagen_en_canvas(pil)
                self._actualizar_stats(caras=self._n_caras_actual, fps=self.fps_actual)
        except Exception:
            pass

    def _detener_webcam(self):
        self.webcam_activa = False

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.detector_webcam:
            try:
                self.detector_webcam.close()
            except Exception:
                pass
            self.detector_webcam = None

        self.btn_webcam.configure(text="▶  ABRIR WEBCAM",
                                  fg_color=COLOR_NARANJA, hover_color=COLOR_NARANJA_HOVER)
        self.btn_imagen.configure(state="normal")
        self._set_status("Webcam detenida")
        self._actualizar_stats(estado="INACTIVO", caras=0, fps=0.0)
        self.lbl_barra_info.configure(text="Sin fuente activa  ")
        self._mostrar_pantalla_inicio()

    # ------------------------------------------
    # MODO IMAGEN
    # ------------------------------------------

    def _cargar_imagen(self):
        if self.webcam_activa:
            self._detener_webcam()

        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Todos", "*.*"),
            ])
        if not ruta:
            return

        self.ruta_imagen_actual = ruta
        nombre = Path(ruta).name
        self._set_status(f"Procesando: {nombre}...", COLOR_AMARILLO)
        self.lbl_barra_info.configure(text=f"IMAGEN: {nombre}  ")
        self.update()

        threading.Thread(target=self._procesar_imagen_thread, args=(ruta,), daemon=True).start()

    def _procesar_imagen_thread(self, ruta: str):
        confianza = self.slider_confianza.get() / 100.0
        img_orig, img_proc, n_caras = procesar_imagen_archivo(ruta, self.modo_blur, confianza)

        if img_orig is None:
            self.after(0, lambda: messagebox.showerror("Error", f"No se pudo cargar:\n{ruta}"))
            return

        orig_pil = Image.fromarray(cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB))
        proc_pil = Image.fromarray(cv2.cvtColor(img_proc, cv2.COLOR_BGR2RGB))

        self.imagen_procesada = proc_pil
        ruta_guardada = self._guardar_automatico(img_proc, ruta)

        self.after(0, lambda: self._mostrar_resultado_imagen(n_caras, ruta_guardada))

    def _mostrar_resultado_imagen(self, n_caras: int, ruta_guardada: str):
        self._mostrar_imagen_en_canvas(self.imagen_procesada)
        self.btn_guardar.configure(state="normal")
        self._actualizar_stats(estado="IMAGEN", caras=n_caras)
        self.lbl_fps.configure(text="N/A")

        if n_caras == 0:
            self._set_status(
                f"Sin rostros detectados — guardado: {Path(ruta_guardada).name}",
                COLOR_TEXTO_GRIS)
        else:
            self._set_status(
                f"{n_caras} rostro(s) detectado(s) — guardado: {Path(ruta_guardada).name}",
                COLOR_EXITO)

    def _guardar_automatico(self, img_bgr: np.ndarray, ruta_original: str) -> str:
        nombre = Path(ruta_original).stem
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        salida = OUTPUT_DIR / f"{nombre}_blurred_{ts}.jpg"
        cv2.imwrite(str(salida), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return str(salida)

    def _guardar_imagen(self):
        if not self.imagen_procesada:
            return
        ruta = filedialog.asksaveasfilename(
            title="Guardar imagen procesada",
            defaultextension=".jpg",
            initialdir=str(OUTPUT_DIR),
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Todos", "*.*")])
        if ruta:
            arr = np.array(self.imagen_procesada)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(ruta, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            self._set_status(f"Guardado: {Path(ruta).name}", COLOR_EXITO)

    # ------------------------------------------
    # CANVAS
    # ------------------------------------------

    def _mostrar_imagen_en_canvas(self, pil_img: Image.Image):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            return

        iw, ih = pil_img.size
        escala  = min(cw / iw, ch / ih)
        nw, nh  = int(iw * escala), int(ih * escala)

        img_r  = pil_img.resize((nw, nh), Image.LANCZOS)
        photo  = ImageTk.PhotoImage(img_r)
        self._photo_actual = photo  # evitar garbage collection

        x = (cw - nw) // 2
        y = (ch - nh) // 2
        self.canvas.delete("all")
        self.canvas.create_image(x, y, anchor="nw", image=photo)

    # ------------------------------------------
    # CONTROLES
    # ------------------------------------------

    def _cambiar_efecto(self, valor: str):
        self.modo_blur = "gaussian" if valor == "GAUSSIANO" else "pixelado"
        self._actualizar_stats(efecto=valor)
        self._set_status(f"Efecto: {valor}")

    def _actualizar_confianza(self, valor):
        self.label_confianza.configure(text=f"Confianza mínima: {int(valor)}%")

    # ------------------------------------------
    # CIERRE
    # ------------------------------------------

    def on_closing(self):
        self.webcam_activa = False
        if self.cap:
            self.cap.release()
        if self.detector_webcam:
            try:
                self.detector_webcam.close()
            except Exception:
                pass
        self.destroy()


# =============================================
# ENTRADA
# =============================================

if __name__ == "__main__":
    app = FaceBlurApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()