import tkinter as tk
from tkinter import filedialog
from enum import Enum
import Viajante_tradicional_optimizado as TSP

# Clases con enumeraciones de los tipos de selección, mutación y crossover
class Seleccion(Enum):
    RULETA_PESOS = 1
    TORNEO = 2

class Mutacion(Enum):
    INTERCAMBIAR_INDICES = 1
    PERMUTAR_ZONA = 2
    INTERCAMBIAR_GENES_VECINOS = 3
    
class Crossover(Enum):
    CROSSOVER_PARTIALLY_MAPPED = 1
    CROSSOVER_ORDER = 2
    CROSSOVER_CYCLE = 3
    EDGE_RECOMBINATION_CROSSOVER = 4
    CROSSOVER_PDF = 5
    
# Hacer un mapeo entre las enumeraciones y las funciones de Viajante_tradicional_optimizado.py
class EnumMapper:
    @staticmethod
    def seleccion(seleccion: Seleccion) -> callable:
        if seleccion == Seleccion.RULETA_PESOS:
            return TSP.seleccionar_ruleta_pesos_optimizado
        elif seleccion == Seleccion.TORNEO:
            return TSP.seleccionar_torneo_optimizado
        else:
            return None

    @staticmethod
    def mutacion(mutacion: Mutacion) -> callable: 
        if mutacion == Mutacion.INTERCAMBIAR_INDICES:
            return TSP.mutar_optimizada
        elif mutacion == Mutacion.PERMUTAR_ZONA:
            return TSP.mutar_desordenado_optimizada
        elif mutacion == Mutacion.INTERCAMBIAR_GENES_VECINOS:
            return TSP.mutar_mejorada_optimizada
        else:
            return None

    @staticmethod
    def crossover(crossover: Crossover) -> callable:
        if crossover == Crossover.CROSSOVER_PARTIALLY_MAPPED:
            return TSP.crossover_partially_mapped_optimizado
        elif crossover == Crossover.CROSSOVER_ORDER:
            return TSP.crossover_order_optimizado
        elif crossover == Crossover.CROSSOVER_CYCLE:
            return TSP.crossover_cycle_optimizado
        elif crossover == Crossover.EDGE_RECOMBINATION_CROSSOVER:
            return TSP.crossover_edge_recombination_optimizado
        elif crossover == Crossover.CROSSOVER_PDF:
            return TSP.crossover_pdf_optimizado
        else:
            return None
    

class GeneticAlgorithmUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Algoritmo Genético")

        # Variables de control
        self.num_ejecuciones = tk.IntVar(value=1)
        self.num_iteraciones = tk.StringVar(value="1000")
        self.prob_mutacion = tk.StringVar(value="0.13")
        self.prob_crossover = tk.StringVar(value="0.35")
        self.num_individuos = tk.StringVar(value="100")
        self.elitismo_var = tk.BooleanVar(value=True)
        self.elitismo_tipo = tk.StringVar(value="Padres vs hijos")
        self.num_padres_pasados = tk.StringVar(value="1")
        self.seleccion_tipo = tk.StringVar(value="Ruleta con pesos")
        self.num_participantes = tk.StringVar(value="2")
        self.usar_biblioteca = tk.BooleanVar(value=False)
        self.mutacion_tipo = tk.StringVar(value="Permutar zona")
        self.crossover_tipo = tk.StringVar(value="Crossover order")

        # Crear elementos de la interfaz
        self.create_widgets()

    def create_widgets(self):
        # Frame para agrupar la configuración principal
        main_frame = tk.Frame(self, bd=2, relief=tk.GROOVE)
        main_frame.grid(row=0, column=0, padx=10, pady=10)

        # Configuración principal
        config_label = tk.Label(main_frame, text="Configuración principal", font=("Helvetica", 12, "bold"))
        config_label.grid(row=0, column=0, columnspan=2, pady=5)

        # Numero de ejecuciones
        ejecuciones_label = tk.Label(main_frame, text="Numero de ejecuciones:")
        ejecuciones_label.grid(row=1, column=0, sticky="w")
        ejecuciones_slider = tk.Scale(main_frame, from_=1, to=16, variable=self.num_ejecuciones, orient="horizontal")
        ejecuciones_slider.grid(row=1, column=1, sticky="we")

        # Numero de iteraciones
        iteraciones_label = tk.Label(main_frame, text="Numero de iteraciones:")
        iteraciones_label.grid(row=2, column=0, sticky="w")
        iteraciones_entry = tk.Entry(main_frame, textvariable=self.num_iteraciones)
        iteraciones_entry.grid(row=2, column=1, sticky="we")

        # Probabilidad de mutacion
        mutacion_label = tk.Label(main_frame, text="Probabilidad de mutacion:")
        mutacion_label.grid(row=3, column=0, sticky="w")
        mutacion_entry = tk.Entry(main_frame, textvariable=self.prob_mutacion)
        mutacion_entry.grid(row=3, column=1, sticky="we")

        # Probabilidad de crossover
        crossover_label = tk.Label(main_frame, text="Probabilidad de crossover:")
        crossover_label.grid(row=4, column=0, sticky="w")
        crossover_entry = tk.Entry(main_frame, textvariable=self.prob_crossover)
        crossover_entry.grid(row=4, column=1, sticky="we")

        # Numero de individuos
        individuos_label = tk.Label(main_frame, text="Numero de individuos:")
        individuos_label.grid(row=5, column=0, sticky="w")
        individuos_entry = tk.Entry(main_frame, textvariable=self.num_individuos)
        individuos_entry.grid(row=5, column=1, sticky="we")

        # Frame para la configuración adicional
        extra_frame = tk.Frame(self, bd=2, relief=tk.GROOVE)
        extra_frame.grid(row=0, column=1, padx=10, pady=10)

        # Configuración adicional
        extra_label = tk.Label(extra_frame, text="Configuración adicional", font=("Helvetica", 12, "bold"))
        extra_label.grid(row=0, column=0, columnspan=2, pady=5)

        # File picker
        file_button = tk.Button(extra_frame, text="Seleccionar fichero", command=self.pick_file)
        file_button.grid(row=1, column=0, columnspan=2, pady=5)

        # Elitismo
        elitismo_check = tk.Checkbutton(extra_frame, text="Elitismo", variable=self.elitismo_var, command=self.toggle_elitismo)
        elitismo_check.grid(row=2, column=0, columnspan=2, sticky="w")
        self.elitismo_widgets = []
        self.elitismo_dropdown = tk.OptionMenu(extra_frame, self.elitismo_tipo, "Padres vs hijos", "Pasar n padres")
        self.elitismo_widgets.append(self.elitismo_dropdown)
        self.elitismo_dropdown.grid(row=3, column=0, sticky="we")
        self.num_padres_entry = tk.Entry(extra_frame, textvariable=self.num_padres_pasados)
        self.elitismo_widgets.append(self.num_padres_entry)
        self.num_padres_entry.grid(row=3, column=1, sticky="we")
        self.toggle_elitismo()

        # Selección
        seleccion_label = tk.Label(extra_frame, text="Selección:")
        seleccion_label.grid(row=4, column=0, sticky="w")
        seleccion_dropdown = tk.OptionMenu(extra_frame, self.seleccion_tipo, *map(lambda x: x.name.title().replace("_", " "), Seleccion))
        seleccion_dropdown.grid(row=4, column=1, sticky="we")

        # Usar biblioteca
        biblioteca_check = tk.Checkbutton(extra_frame, text="Usar biblioteca", variable=self.usar_biblioteca, command=self.toggle_biblioteca)
        biblioteca_check.grid(row=5, column=0, columnspan=2, sticky="w")

        # Mutación
        mutacion_label = tk.Label(extra_frame, text="Mutación:")
        mutacion_label.grid(row=6, column=0, sticky="w")
        mutacion_dropdown = tk.OptionMenu(extra_frame, self.mutacion_tipo, *map(lambda x: x.name.title().replace("_", " "), Mutacion))
        mutacion_dropdown.grid(row=6, column=1, sticky="we")

        # Crossover
        crossover_label = tk.Label(extra_frame, text="Crossover:")
        crossover_label.grid(row=7, column=0, sticky="w")
        crossover_dropdown = tk.OptionMenu(extra_frame, self.crossover_tipo, *map(lambda x: x.name.title().replace("_", " "), Crossover))
        crossover_dropdown.grid(row=7, column=1, sticky="we")

        # Botón ejecutar
        ejecutar_button = tk.Button(self, text="Ejecutar", command=self.ejecutar)
        ejecutar_button.grid(row=1, column=0, columnspan=2, pady=10)

    def toggle_elitismo(self):
        state = "normal" if self.elitismo_var.get() else "disabled"
        for widget in self.elitismo_widgets:
            widget.configure(state=state)

    def toggle_biblioteca(self):
        state = "disabled" if self.usar_biblioteca.get() else "normal"
        self.elitismo_dropdown.configure(state=state)
        self.num_padres_entry.configure(state=state)

    def pick_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        print("Archivo seleccionado:", filename)

    def ejecutar(self):
        # Aquí puedes escribir la lógica para ejecutar el algoritmo genético utilizando los valores proporcionados
        pass

# Inicializar la interfaz
if __name__ == "__main__":
    app = GeneticAlgorithmUI()
    app.mainloop()
