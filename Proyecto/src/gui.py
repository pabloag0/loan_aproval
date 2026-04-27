"""Interfaz grafica para ejecutar el mismo flujo que main.py."""

from contextlib import redirect_stdout
from pathlib import Path
import os
import queue
import sys
import threading
import traceback
import tkinter as tk
from tkinter import messagebox, ttk

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.append(str(PROJECT_DIR))

import main as pipeline
from src import eda


BG = "#eef2f7"
SURFACE = "#ffffff"
INK = "#172033"
MUTED = "#64748b"
ACCENT = "#2563eb"
ACCENT_DARK = "#1d4ed8"
BORDER = "#d8dee9"
LOG_BG = "#111827"
LOG_FG = "#e5e7eb"
SUCCESS = "#15803d"

PHASES = [
    "Cargando datos",
    "Analisis exploratorio",
    "Preprocesado",
    "Regresion logistica",
    "Red original",
    "Datos balanceados",
    "Red balanceada",
    "Red profunda",
    "Evaluacion",
]

METRICS = [
    ("logistica", "Regresion logistica"),
    ("original", "Red original"),
    ("aprobados", "Prestamos aprobados"),
    ("balanceado", "Red balanceada"),
    ("profunda", "Red profunda"),
]


class QueueWriter:
    """Envia la salida de print a la cola de la interfaz."""

    encoding = "utf-8"

    def __init__(self, message_queue):
        self.message_queue = message_queue

    def write(self, text):
        if text:
            self.message_queue.put(("log", text))

    def flush(self):
        pass


class LoanAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisis de Prestamos - Proyecto AA")
        self.root.geometry("1120x760")
        self.root.minsize(980, 650)
        self.root.configure(bg=BG)

        self.message_queue = queue.Queue()
        self.worker = None
        self.running = False
        self.results = {}

        self.run_eda_var = tk.BooleanVar(value=False)
        self.plot_eda_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Listo para ejecutar el analisis.")
        self.progress_var = tk.DoubleVar(value=0)

        self.dataset_vars = {
            "filas": tk.StringVar(value="-"),
            "columnas": tk.StringVar(value="-"),
            "aprobados": tk.StringVar(value="-"),
            "rechazados": tk.StringVar(value="-"),
        }
        self.metric_vars = {
            key: tk.StringVar(value="--")
            for key, _ in METRICS
        }
        self.phase_labels = {}

        self.configure_styles()
        self.create_widgets()
        self.sync_eda_controls()
        self.append_log("Listo. Configura la ejecucion y pulsa Iniciar analisis.\n")

    def configure_styles(self):
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.style.configure("App.TFrame", background=BG)
        self.style.configure("Card.TFrame", background=SURFACE)
        self.style.configure("Header.TLabel", background=BG, foreground=INK, font=("Segoe UI", 22, "bold"))
        self.style.configure("Subheader.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        self.style.configure("CardTitle.TLabel", background=SURFACE, foreground=INK, font=("Segoe UI", 12, "bold"))
        self.style.configure("Body.TLabel", background=SURFACE, foreground=MUTED, font=("Segoe UI", 10))
        self.style.configure("Value.TLabel", background=SURFACE, foreground=INK, font=("Segoe UI", 14, "bold"))
        self.style.configure("Metric.TLabel", background=SURFACE, foreground=ACCENT, font=("Segoe UI", 20, "bold"))
        self.style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))
        self.style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=(10, 7))
        self.style.configure("Accent.Horizontal.TProgressbar", troughcolor="#dbeafe", background=ACCENT)
        self.style.map(
            "Primary.TButton",
            background=[("active", ACCENT_DARK), ("!disabled", ACCENT)],
            foreground=[("!disabled", "#ffffff")],
        )

    def create_widgets(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        app = ttk.Frame(self.root, style="App.TFrame", padding=22)
        app.grid(row=0, column=0, sticky="nsew")
        app.columnconfigure(0, weight=1)
        app.rowconfigure(1, weight=1)

        self.create_header(app)
        self.create_body(app)

    def create_header(self, parent):
        header = ttk.Frame(parent, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Analisis de prestamos", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Ejecuta carga de datos, EDA opcional, preprocesado, entrenamiento y evaluacion desde una sola ventana.",
            style="Subheader.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

    def create_body(self, parent):
        body = ttk.Frame(parent, style="App.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=0, minsize=310)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 18))

        right = ttk.Frame(body, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        self.create_controls(left)
        self.create_dataset_panel(left)
        self.create_phase_panel(left)
        self.create_results_panel(right)
        self.create_progress_panel(right)
        self.create_log_panel(right)

    def create_card(self, parent, padding=16):
        card = tk.Frame(
            parent,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=padding,
            pady=padding,
        )
        return card

    def create_controls(self, parent):
        card = self.create_card(parent)
        card.pack(fill="x", pady=(0, 14))

        tk.Label(card, text="Configuracion", bg=SURFACE, fg=INK, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(
            card,
            text="Elige si quieres incluir el analisis exploratorio antes de entrenar.",
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9),
            wraplength=255,
            justify="left",
        ).pack(anchor="w", pady=(4, 12))

        self.eda_check = ttk.Checkbutton(
            card,
            text="Ejecutar EDA",
            variable=self.run_eda_var,
            command=self.sync_eda_controls,
        )
        self.eda_check.pack(anchor="w", pady=(0, 6))

        self.plot_check = ttk.Checkbutton(
            card,
            text="Generar graficas EDA",
            variable=self.plot_eda_var,
        )
        self.plot_check.pack(anchor="w", pady=(0, 14))

        self.run_button = ttk.Button(
            card,
            text="Iniciar analisis",
            style="Primary.TButton",
            command=self.run_analysis,
        )
        self.run_button.pack(fill="x", pady=(0, 8))

        self.clear_button = ttk.Button(
            card,
            text="Limpiar resultados",
            style="Secondary.TButton",
            command=self.clear_results,
        )
        self.clear_button.pack(fill="x", pady=(0, 8))

        ttk.Button(
            card,
            text="Abrir carpeta results",
            style="Secondary.TButton",
            command=self.open_results_folder,
        ).pack(fill="x")

    def create_dataset_panel(self, parent):
        card = self.create_card(parent)
        card.pack(fill="x", pady=(0, 14))

        tk.Label(card, text="Dataset", bg=SURFACE, fg=INK, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.create_stat_row(card, "Filas", self.dataset_vars["filas"])
        self.create_stat_row(card, "Columnas", self.dataset_vars["columnas"])
        self.create_stat_row(card, "Loan status = 1", self.dataset_vars["aprobados"])
        self.create_stat_row(card, "Loan status = 0", self.dataset_vars["rechazados"])

    def create_stat_row(self, parent, label, variable):
        row = tk.Frame(parent, bg=SURFACE)
        row.pack(fill="x", pady=(10, 0))
        tk.Label(row, text=label, bg=SURFACE, fg=MUTED, font=("Segoe UI", 9)).pack(side="left")
        tk.Label(row, textvariable=variable, bg=SURFACE, fg=INK, font=("Segoe UI", 10, "bold")).pack(side="right")

    def create_phase_panel(self, parent):
        card = self.create_card(parent)
        card.pack(fill="x")

        tk.Label(card, text="Flujo", bg=SURFACE, fg=INK, font=("Segoe UI", 12, "bold")).pack(anchor="w")
        tk.Label(
            card,
            text="Las fases se marcan al avanzar el pipeline.",
            bg=SURFACE,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", pady=(4, 10))

        for phase in PHASES:
            label = tk.Label(
                card,
                text=f"[ ] {phase}",
                bg=SURFACE,
                fg=MUTED,
                font=("Segoe UI", 9),
                anchor="w",
            )
            label.pack(fill="x", pady=2)
            self.phase_labels[phase] = label

    def create_results_panel(self, parent):
        panel = ttk.Frame(parent, style="App.TFrame")
        panel.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        for column in range(3):
            panel.columnconfigure(column, weight=1, uniform="metrics")

        for index, (key, title) in enumerate(METRICS):
            row = index // 3
            column = index % 3
            card = self.create_metric_card(panel, title, self.metric_vars[key])
            card.grid(row=row, column=column, sticky="ew", padx=(0 if column == 0 else 10, 0), pady=(0, 10))

    def create_metric_card(self, parent, title, variable):
        card = tk.Frame(
            parent,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=16,
            pady=14,
        )
        tk.Label(card, text=title, bg=SURFACE, fg=MUTED, font=("Segoe UI", 9)).pack(anchor="w")
        tk.Label(card, textvariable=variable, bg=SURFACE, fg=ACCENT, font=("Segoe UI", 20, "bold")).pack(anchor="w")
        return card

    def create_progress_panel(self, parent):
        card = self.create_card(parent, padding=14)
        card.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        card.columnconfigure(0, weight=1)

        tk.Label(card, textvariable=self.status_var, bg=SURFACE, fg=INK, font=("Segoe UI", 10, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.progress = ttk.Progressbar(
            card,
            variable=self.progress_var,
            maximum=100,
            style="Accent.Horizontal.TProgressbar",
        )
        self.progress.grid(row=1, column=0, sticky="ew", pady=(10, 0))

    def create_log_panel(self, parent):
        card = self.create_card(parent, padding=0)
        card.grid(row=2, column=0, sticky="nsew")
        card.columnconfigure(0, weight=1)
        card.rowconfigure(1, weight=1)

        header = tk.Frame(card, bg=SURFACE, padx=16, pady=12)
        header.grid(row=0, column=0, sticky="ew")
        tk.Label(header, text="Salida del pipeline", bg=SURFACE, fg=INK, font=("Segoe UI", 12, "bold")).pack(side="left")
        tk.Label(header, text="En vivo", bg=SURFACE, fg=SUCCESS, font=("Segoe UI", 9, "bold")).pack(side="right")

        log_wrap = tk.Frame(card, bg=LOG_BG)
        log_wrap.grid(row=1, column=0, sticky="nsew")
        log_wrap.columnconfigure(0, weight=1)
        log_wrap.rowconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_wrap,
            bg=LOG_BG,
            fg=LOG_FG,
            insertbackground=LOG_FG,
            relief="flat",
            bd=0,
            padx=14,
            pady=14,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
        )
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_wrap, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def sync_eda_controls(self):
        if self.run_eda_var.get():
            self.plot_check.state(["!disabled"])
        else:
            self.plot_eda_var.set(False)
            self.plot_check.state(["disabled"])

    def run_analysis(self):
        if self.running:
            return

        self.running = True
        self.results = {}
        self.reset_display()
        self.run_button.configure(state="disabled", text="Analizando...")
        self.clear_button.configure(state="disabled")
        self.eda_check.state(["disabled"])
        self.plot_check.state(["disabled"])

        run_eda = self.run_eda_var.get()
        plot_eda = self.plot_eda_var.get()
        self.worker = threading.Thread(
            target=self.perform_analysis,
            args=(run_eda, plot_eda),
            daemon=True,
        )
        self.worker.start()
        self.root.after(80, self.process_messages)

    def perform_analysis(self, run_eda, plot_eda):
        writer = QueueWriter(self.message_queue)

        try:
            with redirect_stdout(writer):
                self.emit_phase("Cargando datos", 5)
                df = pipeline.cargar_datos()
                self.emit_dataset_stats(df)

                if run_eda:
                    self.emit_phase("Analisis exploratorio", 14)
                    print("Ejecutando analisis exploratorio...")
                    if plot_eda:
                        print("Generando graficas EDA...")
                    eda.show_dataset_info(df, plot=plot_eda)
                else:
                    print("Analisis exploratorio omitido desde la interfaz.")

                self.emit_phase("Preprocesado", 25)
                X_train, X_test, y_train, y_test = pipeline.preprocesar_datos(df)

                self.emit_phase("Regresion logistica", 38)
                accuracy_lr = pipeline.entrenar_regresion_logistica(df)

                self.emit_phase("Red original", 50)
                theta1, theta2 = pipeline.entrenar_red_original(X_train, y_train)

                self.emit_phase("Datos balanceados", 60)
                X_train_bal, X_test_bal, y_train_bal, y_test_bal = pipeline.preprocesar_datos_balanceados(df)

                self.emit_phase("Red balanceada", 72)
                theta1_bal, theta2_bal = pipeline.entrenar_red_balanceada(X_train_bal, y_train_bal)

                self.emit_phase("Red profunda", 84)
                _, metricas_dnn = pipeline.entrenar_red_profunda(
                    X_train_bal,
                    X_test_bal,
                    y_train_bal,
                    y_test_bal,
                )

                self.emit_phase("Evaluacion", 95)
                resultados = {
                    "logistica": accuracy_lr,
                    "original": pipeline.evaluar_red(theta1, theta2, X_test, y_test),
                    "aprobados": pipeline.evaluar_prestamos_aprobados(df, theta1, theta2),
                    "balanceado": pipeline.evaluar_red(theta1_bal, theta2_bal, X_test_bal, y_test_bal),
                    "profunda": metricas_dnn["accuracy"] * 100,
                }

                pipeline.mostrar_resultados(resultados)
                self.message_queue.put(("done", resultados))

        except Exception:
            self.message_queue.put(("error", traceback.format_exc()))

    def emit_phase(self, phase, progress):
        self.message_queue.put(("phase", phase))
        self.message_queue.put(("status", phase))
        self.message_queue.put(("progress", progress))

    def emit_dataset_stats(self, df):
        status_counts = df["loan_status"].value_counts() if "loan_status" in df.columns else {}
        stats = {
            "filas": len(df),
            "columnas": len(df.columns),
            "aprobados": int(status_counts.get(1, 0)),
            "rechazados": int(status_counts.get(0, 0)),
        }
        self.message_queue.put(("dataset", stats))

    def process_messages(self):
        try:
            while True:
                kind, payload = self.message_queue.get_nowait()

                if kind == "log":
                    self.append_log(payload)
                elif kind == "phase":
                    self.update_phase(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "progress":
                    self.progress_var.set(payload)
                elif kind == "dataset":
                    self.update_dataset(payload)
                elif kind == "done":
                    self.finish_success(payload)
                elif kind == "error":
                    self.finish_error(payload)

        except queue.Empty:
            pass

        if self.running:
            self.root.after(80, self.process_messages)

    def append_log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def update_dataset(self, stats):
        for key, value in stats.items():
            self.dataset_vars[key].set(str(value))

    def update_phase(self, active_phase):
        active_index = PHASES.index(active_phase)

        for index, phase in enumerate(PHASES):
            label = self.phase_labels[phase]
            if index < active_index:
                label.configure(text=f"[x] {phase}", fg=SUCCESS, font=("Segoe UI", 9))
            elif index == active_index:
                label.configure(text=f"> {phase}", fg=ACCENT, font=("Segoe UI", 9, "bold"))
            else:
                label.configure(text=f"[ ] {phase}", fg=MUTED, font=("Segoe UI", 9))

    def finish_success(self, resultados):
        self.results = resultados
        for key, value in resultados.items():
            self.metric_vars[key].set(f"{value:.2f}%")

        for phase in PHASES:
            self.phase_labels[phase].configure(text=f"[x] {phase}", fg=SUCCESS, font=("Segoe UI", 9))

        self.progress_var.set(100)
        self.status_var.set("Analisis completado.")
        self.finish_run()

    def finish_error(self, error_text):
        self.append_log("\nERROR DURANTE EL ANALISIS\n")
        self.append_log(error_text)
        self.status_var.set("Error durante el analisis.")
        self.progress_var.set(0)
        self.finish_run()
        messagebox.showerror("Error", "El analisis no pudo completarse. Revisa la salida del pipeline.")

    def finish_run(self):
        self.running = False
        self.run_button.configure(state="normal", text="Iniciar analisis")
        self.clear_button.configure(state="normal")
        self.eda_check.state(["!disabled"])
        self.sync_eda_controls()

    def reset_display(self):
        for variable in self.metric_vars.values():
            variable.set("--")

        for variable in self.dataset_vars.values():
            variable.set("-")

        for phase in PHASES:
            self.phase_labels[phase].configure(text=f"[ ] {phase}", fg=MUTED, font=("Segoe UI", 9))

        self.progress_var.set(0)
        self.status_var.set("Preparando ejecucion...")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.insert("end", "Listo. La salida del entrenamiento aparecera aqui.\n\n")
        self.log_text.configure(state="disabled")

    def clear_results(self):
        if self.running:
            return

        self.results = {}
        self.reset_display()
        self.status_var.set("Resultados limpiados.")

    def open_results_folder(self):
        results_dir = PROJECT_DIR / "results"
        results_dir.mkdir(exist_ok=True)

        try:
            os.startfile(str(results_dir))
        except (AttributeError, OSError):
            messagebox.showwarning("Carpeta no disponible", f"No se pudo abrir:\n{results_dir}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LoanAnalysisGUI(root)
    root.mainloop()
