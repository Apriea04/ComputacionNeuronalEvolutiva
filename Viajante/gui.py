import tkinter as tk
from tkinter import filedialog
from enums import Seleccion, Mutacion, Crossover, Elitismo


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
        self.elitismo_tipo = tk.StringVar(
            value=Elitismo.PASAR_N_PADRES.name.lower().capitalize().replace("_", " ")
        )
        self.num_padres_pasados = tk.StringVar(value="1")
        self.num_padres_pasados_activo = True
        self.seleccion_tipo = tk.StringVar(
            value=Seleccion.TORNEO.name.lower().capitalize().replace("_", " ")
        )
        self.num_participantes = tk.StringVar(value="2")
        self.usar_biblioteca = tk.BooleanVar(value=False)
        self.mutacion_tipo = tk.StringVar(
            value=Mutacion.PERMUTAR_ZONA.name.lower().capitalize().replace("_", " ")
        )
        self.crossover_tipo = tk.StringVar(
            value=Crossover.CROSSOVER_ORDER.name.lower().capitalize().replace("_", " ")
        )
        self.fichero_coordenadas = tk.StringVar()

        # Crear elementos de la interfaz
        self.create_widgets()

    def _enums_to_string(self, enum):
        return enum.name.lower().capitalize().replace("_", " ")

    def create_widgets(self):
        # Frame para agrupar la configuración principal
        main_frame = tk.Frame(self, bd=2, relief=tk.GROOVE)
        main_frame.grid(row=0, column=0, padx=10, pady=10)

        # Configuración principal
        config_label = tk.Label(
            main_frame, text="Configuración principal", font=("Helvetica", 12, "bold")
        )
        config_label.grid(row=0, column=0, columnspan=2, pady=5)

        # Numero de ejecuciones
        ejecuciones_label = tk.Label(main_frame, text="Numero de ejecuciones:")
        ejecuciones_label.grid(row=1, column=0, sticky="w")
        ejecuciones_slider = tk.Scale(
            main_frame,
            from_=1,
            to=16,
            variable=self.num_ejecuciones,
            orient="horizontal",
        )
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

        file_frame = tk.Frame(self, bd=2, relief=tk.GROOVE)
        file_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        file_frame_title = tk.Label(
            file_frame, text="Fichero de coordenadas", font=("Helvetica", 12, "bold")
        )
        file_frame_title.grid(row=0, column=0, columnspan=2, pady=5)

        # File picker
        file_button = tk.Button(
            file_frame, text="Seleccionar coordenadas", command=self.pick_file
        )
        file_button.grid(row=1, column=0, columnspan=2, pady=5)

        file_label = tk.Label(file_frame, text="Fichero seleccionado: ")
        file_label.grid(row=2, column=0, columnspan=2, pady=5)
        selected_file_label = tk.Label(
            file_frame, textvariable=self.fichero_coordenadas
        )
        selected_file_label.grid(row=3, columnspan=2)

        # Frame para la configuración adicional
        extra_frame = tk.Frame(self, bd=3, relief=tk.GROOVE)
        extra_frame.grid(row=0, column=1, padx=10, pady=10)

        # Configuración adicional
        extra_label = tk.Label(
            extra_frame, text="Configuración adicional", font=("Helvetica", 12, "bold")
        )
        extra_label.grid(row=0, column=0, columnspan=2, pady=5)

        # Elitismo
        elitismo_check = tk.Checkbutton(
            extra_frame,
            text="Elitismo",
            variable=self.elitismo_var,
            command=self.toggle_elitismo,
        )
        elitismo_check.grid(row=2, column=0, columnspan=2, sticky="w")
        self.elitismo_widgets = []
        self.elitismo_dropdown = tk.OptionMenu(
            extra_frame,
            self.elitismo_tipo,
            *map(self._enums_to_string, Elitismo),
            command=self.seleccion_elitismo,
        )
        self.elitismo_widgets.append(self.elitismo_dropdown)
        self.elitismo_dropdown.grid(row=3, column=0, sticky="we")
        self.num_padres_entry = tk.Entry(
            extra_frame, textvariable=self.num_padres_pasados
        )
        self.elitismo_widgets.append(self.num_padres_entry)
        self.num_padres_entry.grid(row=3, column=1, sticky="we")
        self.toggle_elitismo()

        # Selección
        seleccion_label = tk.Label(extra_frame, text="Selección:")
        seleccion_label.grid(row=4, column=0, sticky="w")
        seleccion_dropdown = tk.OptionMenu(
            extra_frame, self.seleccion_tipo, *map(self._enums_to_string, Seleccion)
        )
        seleccion_dropdown.grid(row=4, column=1, sticky="we")

        # Usar biblioteca
        biblioteca_check = tk.Checkbutton(
            extra_frame,
            text="Usar biblioteca",
            variable=self.usar_biblioteca,
            command=self.toggle_biblioteca,
        )
        biblioteca_check.grid(row=5, column=0, columnspan=2, sticky="w")

        # Mutación
        mutacion_label = tk.Label(extra_frame, text="Mutación:")
        mutacion_label.grid(row=6, column=0, sticky="w")
        self.mutacion_dropdown = tk.OptionMenu(
            extra_frame, self.mutacion_tipo, *map(self._enums_to_string, Mutacion)
        )
        self.mutacion_dropdown.grid(row=6, column=1, sticky="we")

        # Crossover
        crossover_label = tk.Label(extra_frame, text="Crossover:")
        crossover_label.grid(row=7, column=0, sticky="w")
        self.crossover_dropdown = tk.OptionMenu(
            extra_frame, self.crossover_tipo, *map(self._enums_to_string, Crossover)
        )
        self.crossover_dropdown.grid(row=7, column=1, sticky="we")

        # Botón ejecutar
        ejecutar_button = tk.Button(self, text="Ejecutar", command=self.ejecutar)
        ejecutar_button.grid(row=2, column=0, columnspan=2, pady=10)

    def toggle_elitismo(self):
        state = "normal" if self.elitismo_var.get() else "disabled"
        if not self.usar_biblioteca.get():
            for widget in self.elitismo_widgets:
                widget.configure(state=state)
                if state == "normal" and widget is self.num_padres_entry:
                    if self.num_padres_pasados_activo:
                        widget.configure(state="normal")
                    else:
                        widget.configure(state="disabled")

    def toggle_biblioteca(self):
        state = "disabled" if self.usar_biblioteca.get() else "normal"
        self.elitismo_dropdown.configure(state=state)

        # De mutaciones quitamos la opción de intercambiar indices vecinos
        if self.usar_biblioteca.get():
            self.mutacion_dropdown["menu"].delete(2)
            self.num_padres_entry.configure(state="disabled")
            # Eliminamos los crossover que no estén implementados
            self.crossover_dropdown["menu"].delete(4)
            self.crossover_dropdown["menu"].delete(3)
            self.crossover_dropdown["menu"].delete(2)
            
            # Si está seleccionado alguno de los que no hay, lo cambiamos a uno que sí
            if self.crossover_tipo.get() == self._enums_to_string(Crossover.CROSSOVER_CYCLE) or self.crossover_tipo.get() == self._enums_to_string(Crossover.EDGE_RECOMBINATION_CROSSOVER) or self.crossover_tipo.get() == self._enums_to_string(Crossover.CROSSOVER_PDF):
                self.crossover_tipo.set(self._enums_to_string(Crossover.CROSSOVER_ORDER))
                
            if self.mutacion_tipo.get() == self._enums_to_string(Mutacion.INTERCAMBIAR_GENES_VECINOS):
                self.mutacion_tipo.set(self._enums_to_string(Mutacion.PERMUTAR_ZONA))
                
            if self.elitismo_tipo.get() == self._enums_to_string(Elitismo.PASAR_N_PADRES):
                self.elitismo_tipo.set(self._enums_to_string(Elitismo.PADRES_VS_HIJOS))
        else:
            if self.num_padres_pasados_activo:
                self.num_padres_entry.configure(state="normal")
            # Insertamos la mutacion en el desplegable
            self.mutacion_dropdown["menu"].insert_command(
                2,
                label=self._enums_to_string(Mutacion.INTERCAMBIAR_GENES_VECINOS),
                command=lambda: self.mutacion_tipo.set(self._enums_to_string(Mutacion.INTERCAMBIAR_GENES_VECINOS)),
            )

            # Insertamos los crossover que hemos quitado
            self.crossover_dropdown["menu"].insert_command(
                3,
                label=self._enums_to_string(Crossover.CROSSOVER_CYCLE),
                command=lambda: self.crossover_tipo.set(
                    self._enums_to_string(Crossover.CROSSOVER_CYCLE)
                ),
            )
            self.crossover_dropdown["menu"].insert_command(
                4,
                label=self._enums_to_string(Crossover.EDGE_RECOMBINATION_CROSSOVER),
                command=lambda: self.crossover_tipo.set(
                    self._enums_to_string(Crossover.EDGE_RECOMBINATION_CROSSOVER)
                ),
            )
            self.crossover_dropdown["menu"].insert_command(
                5,
                label=self._enums_to_string(Crossover.CROSSOVER_PDF),
                command=lambda: self.crossover_tipo.set(
                    self._enums_to_string(Crossover.CROSSOVER_PDF)
                ),
            )

    def pick_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        self.fichero_coordenadas.set(filename)
        print("Archivo seleccionado:", filename)

    def ejecutar(self):
        # Aquí puedes escribir la lógica para ejecutar el algoritmo genético utilizando los valores proporcionados
        pass

    def seleccion_elitismo(self, value):
        # Si se selecciona elitismo N Padres, se debe activar el número de padres a pasar. Hacer el mapeo con la enumeración
        if value == self._enums_to_string(Elitismo.PASAR_N_PADRES):
            self.num_padres_entry.configure(state="normal")
            self.num_padres_pasados_activo = True
        else:
            self.num_padres_entry.configure(state="disabled")
            self.num_padres_pasados_activo = False


# Inicializar la interfaz
if __name__ == "__main__":
    app = GeneticAlgorithmUI()
    app.mainloop()
