import tkinter as tk

class Spinner(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Spinner")
        self.geometry("200x200")
        self.configure(bg="white")

        self.canvas = tk.Canvas(self, width=100, height=100, bg="white", highlightthickness=0)
        self.canvas.pack(pady=50)

        self.spinner = self.canvas.create_arc(10, 10, 90, 90, start=0, extent=30, outline="black", width=6, style=tk.ARC)
        self.direction = 1  # Dirección de crecimiento del segmento
        self.length = 50     # Longitud inicial del segmento
        self.animate_spinner()

    def animate_spinner(self, start=0):
        self.canvas.itemconfigure(self.spinner, start=start, extent=self.length)
        start += 5
        # Ajustar la longitud del segmento
        self.length += self.direction * 2
        if self.length >= 120 or self.length <= 10:
            # Cambiar la dirección cuando alcance los límites
            self.direction *= -1
        self.after(25, lambda: self.animate_spinner(start))

if __name__ == "__main__":
    app = Spinner()
    app.mainloop()
