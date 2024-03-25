import tkinter as tk
import threading
import time

class SpinnerModal(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.geometry("200x200")
        self.title("Spinner Modal")

        self.canvas = tk.Canvas(self, width=100, height=100, bg="white", highlightthickness=0)
        self.canvas.pack(pady=50)

        self.spinner = self.canvas.create_arc(10, 10, 90, 90, start=0, extent=30, outline="black", width=6, style=tk.ARC)
        self.direction = 1  
        self.length = 50     
        self.animate_spinner()

    def animate_spinner(self, start=0):
        self.canvas.itemconfigure(self.spinner, start=start, extent=self.length)
        start += 5
        self.length += self.direction * 2
        if self.length >= 120 or self.length <= 10:
            self.direction *= -1
        self.after(25, lambda: self.animate_spinner(start))

class MyGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Main Window")
        self.geometry("200x200")

        self.button = tk.Button(self, text="Ejecutar", command=self.ejecutar)
        self.button.pack(pady=20)

        self.spinner_modal = None
        self.work_finished = threading.Condition()  # Variable de condición

    def ejecutar(self):
        # Mostrar el spinner modal
        self.spinner_modal = SpinnerModal(self)
        self.spinner_modal.grab_set()  # Bloquear el foco de la ventana principal
        self.spinner_modal.protocol("WM_DELETE_WINDOW", self.on_spinner_close)

        # Crear un hilo para ejecutar el trabajo
        worker_thread = threading.Thread(target=self._execute_code)
        worker_thread.start()

    def _execute_code(self):
        # Simular una tarea larga
        time.sleep(5)

        # Cuando el trabajo está hecho, notificar a la ventana principal
        with self.work_finished:
            self.work_finished.notify()

    def on_spinner_close(self):
        # Ignorar el cierre de la ventana modal mientras el trabajo está en curso
        pass

if __name__ == "__main__":
    app = MyGUI()
    app.mainloop()
