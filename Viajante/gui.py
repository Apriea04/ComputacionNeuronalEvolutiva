from os import system
import os
from signal import signal
import threading
import tkinter as tk
from tkinter import filedialog
from enums import Seleccion, Mutacion, Crossover, Elitismo
import TSP_chapa as miTSP
import TSP_biblioteca as deapTSP
import subprocess
import ctypes
import signal

class GeneticAlgorithmUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Configuración del Algoritmo Genético TSP")

        # Variables de control
        self.num_ejecuciones = tk.IntVar(value=2)
        self.num_iteraciones = tk.IntVar(value=1000)
        self.prob_mutacion = tk.DoubleVar(value=0.13)
        self.prob_crossover = tk.DoubleVar(value=0.35)
        self.num_individuos = tk.IntVar(value=100)
        self.elitismo_var = tk.BooleanVar(value=True)
        self.elitismo_tipo = tk.StringVar(
            value=Elitismo.PASAR_N_PADRES.name.lower().capitalize().replace("_", " ")
        )
        self.num_padres_pasados = tk.IntVar(value=3)
        self.num_padres_pasados_activo = True
        self.seleccion_tipo = tk.StringVar(
            value=Seleccion.TORNEO.name.lower().capitalize().replace("_", " ")
        )
        self.participantes_torneo = tk.IntVar(value=3)
        self.usar_biblioteca = tk.BooleanVar(value=False)
        self.mutacion_tipo = tk.StringVar(
            value=Mutacion.PERMUTAR_ZONA.name.lower().capitalize().replace("_", " ")
        )
        self.crossover_tipo = tk.StringVar(
            value=Crossover.CROSSOVER_ORDER.name.lower().capitalize().replace("_", " ")
        )
        self.fichero_coordenadas = tk.StringVar(value="Ningún archivo seleccionado")
        self.verbose_var = tk.BooleanVar(value=True)

        # Varibles de control para el algoritmo sin biblioteca
        self.dibujar_evolucion = tk.BooleanVar(value=False)
        self.parada_en_media = tk.BooleanVar(value=True)
        self.max_medias_iguales = tk.IntVar(value=10)
        self.parada_en_clones = tk.BooleanVar(value=False)
        self.plot_resultados_parciales = tk.BooleanVar(value=False)
        self.cambio_de_mutacion = tk.BooleanVar(value=False)

        # Multiprocesamiento TODO
        self.hilos = []

        # Crear elementos de la interfaz
        self.create_widgets()

    def _enums_to_string(self, enum):
        return enum.name.lower().capitalize().replace("_", " ")

    def create_widgets(self):
        # Frame para agrupar la configuración principal
        main_frame = tk.Frame(self, bd=3, relief=tk.GROOVE)
        main_frame.grid(row=0, column=0, padx=10, pady=10)

        # Configuración principal
        config_label = tk.Label(
            main_frame, text="Configuración principal", font=("Helvetica", 12, "bold")
        )
        config_label.grid(row=0, column=0, columnspan=2, pady=5)

        # Numero de ejecuciones
        ejecuciones_label = tk.Label(
            main_frame, text="Número de ejecuciones paralelas:"
        )
        ejecuciones_label.grid(row=1, column=0, sticky="w")
        ejecuciones_slider = tk.Scale(
            main_frame,
            from_=1,
            to=os.cpu_count()*2,
            variable=self.num_ejecuciones,
            orient="horizontal",
        )
        ejecuciones_slider.grid(row=1, column=1, sticky="we")

        # Numero de iteraciones
        iteraciones_label = tk.Label(main_frame, text="Número de iteraciones:")
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
        individuos_label = tk.Label(main_frame, text="Número de individuos:")
        individuos_label.grid(row=5, column=0, sticky="w")
        individuos_entry = tk.Entry(main_frame, textvariable=self.num_individuos)
        individuos_entry.grid(row=5, column=1, sticky="we")

        # Verbose
        verbose_check = tk.Checkbutton(
            main_frame, text="Verbose", variable=self.verbose_var
        )
        verbose_check.grid(row=6, column=0, columnspan=1, sticky="w")

        # Usar biblioteca
        biblioteca_check = tk.Checkbutton(
            main_frame,
            text="Usar biblioteca",
            variable=self.usar_biblioteca,
            command=self.toggle_biblioteca,
            font=("Helvetica", 9, "bold")
        )
        biblioteca_check.grid(row=6, column=1, columnspan=1, sticky="w")

        file_frame = tk.Frame(self, bd=3, relief=tk.GROOVE)
        file_frame.grid(row=1, column=1, columnspan=2, padx=10, pady=10)

        file_frame_title = tk.Label(
            file_frame, text="Fichero de coordenadas", font=("Helvetica", 12, "bold")
        )
        file_frame_title.grid(row=0, column=0, columnspan=2, pady=5)

        # File picker
        file_button = tk.Button(
            file_frame, text="Seleccionar coordenadas", command=self.pick_file
        )
        file_button.grid(row=1, column=0, columnspan=2, pady=5)
        selected_file_label = tk.Label(
            file_frame, textvariable=self.fichero_coordenadas
        )
        selected_file_label.grid(row=2, columnspan=2)

        # Frame para la configuración adicional
        operadores_frame = tk.Frame(self, bd=3, relief=tk.GROOVE)
        operadores_frame.grid(row=0, column=1, padx=10, pady=10)

        # Configuración adicional
        operadores_label = tk.Label(
            operadores_frame,
            text="Selección de operadores",
            font=("Helvetica", 12, "bold"),
        )
        operadores_label.grid(row=0, column=0, columnspan=2, pady=5)

        # Elitismo
        elitismo_frame = tk.Frame(operadores_frame, bd=1, relief=tk.SOLID)
        elitismo_frame.grid(row=1, column=0, columnspan=2, pady=5, padx=5)

        self.elitismo_check = tk.Checkbutton(
            elitismo_frame,
            text="Elitismo",
            variable=self.elitismo_var,
            command=self.toggle_elitismo,
        )
        self.elitismo_check.grid(row=2, column=0, columnspan=2, sticky="w")
        self.elitismo_widgets = []
        self.elitismo_dropdown = tk.OptionMenu(
            elitismo_frame,
            self.elitismo_tipo,
            *map(self._enums_to_string, Elitismo),
            command=self.seleccion_elitismo,
        )
        self.elitismo_widgets.append(self.elitismo_dropdown)
        self.elitismo_dropdown.grid(row=3, column=0, sticky="we")
        self.num_padres_entry = tk.Entry(
            elitismo_frame, textvariable=self.num_padres_pasados
        )
        self.elitismo_widgets.append(self.num_padres_entry)
        self.num_padres_entry.grid(row=3, column=1, sticky="we")
        self.toggle_elitismo()

        # Selección
        seleccion_frame = tk.Frame(operadores_frame, bd=1, relief=tk.SOLID)
        seleccion_frame.grid(row=4, column=0, columnspan=2, pady=5, padx=5, sticky="we")
        seleccion_label = tk.Label(seleccion_frame, text="Selección:")
        seleccion_label.grid(row=4, column=0, sticky="w")
        seleccion_dropdown = tk.OptionMenu(
            seleccion_frame,
            self.seleccion_tipo,
            *map(self._enums_to_string, Seleccion),
            command=self.toggle_seleccion,
        )
        seleccion_dropdown.grid(row=4, column=1, sticky="we")
        self.participantes_torneo_label = tk.Label(
            seleccion_frame, text="Participantes torneo:"
        )
        self.participantes_torneo_label.grid(row=5, column=0, sticky="w")
        self.participantes_torneo_entry = tk.Entry(
            seleccion_frame, textvariable=self.participantes_torneo
        )
        self.participantes_torneo_entry.grid(row=5, column=1, sticky="we")

        # Mutación
        mutacion_label = tk.Label(operadores_frame, text="Mutación:")
        mutacion_label.grid(row=6, column=0, sticky="w")
        self.mutacion_dropdown = tk.OptionMenu(
            operadores_frame, self.mutacion_tipo, *map(self._enums_to_string, Mutacion)
        )
        self.mutacion_dropdown.grid(row=6, column=1, sticky="we")

        # Crossover
        crossover_label = tk.Label(operadores_frame, text="Crossover:")
        crossover_label.grid(row=7, column=0, sticky="w")
        self.crossover_dropdown = tk.OptionMenu(
            operadores_frame,
            self.crossover_tipo,
            *map(self._enums_to_string, Crossover),
        )
        self.crossover_dropdown.grid(row=7, column=1, sticky="we")

        # Botón ejecutar
        self.ejecutar_button = tk.Button(self, text="Ejecutar", command=self.ejecutar)
        self.ejecutar_button.grid(row=3, column=0, columnspan=2, pady=10)

        # Frame para configuración específica del algoritmo sin biblioteca
        self.sin_biblioteca_frame = tk.Frame(self, bd=3, relief=tk.GROOVE)
        self.sin_biblioteca_frame.grid(row=1, column=0, columnspan=1, padx=10, pady=10)

        sin_biblioteca_title = tk.Label(
            self.sin_biblioteca_frame,
            text="Configuración específica del algoritmo sin biblioteca",
            font=("Helvetica", 12, "bold"),
        )

        sin_biblioteca_title.grid(row=0, column=0, columnspan=2, pady=5)

        # Dibujar evolución
        dibujar_evolucion_check = tk.Checkbutton(
            self.sin_biblioteca_frame,
            text="Mostrar gráfica con la evolución",
            variable=self.dibujar_evolucion,
        )
        dibujar_evolucion_check.grid(row=1, column=0, columnspan=1, sticky="w")

        # Cambios en media
        acciones_media_frame = tk.Frame(
            self.sin_biblioteca_frame, bd=1, relief=tk.SOLID
        )
        acciones_media_frame.grid(
            row=3, column=0, columnspan=2, pady=5, padx=5, sticky="we"
        )

        parada_en_media_label = tk.Label(
            self.sin_biblioteca_frame, text="Acciones si la media no mejora:"
        )
        parada_en_media_label.grid(row=2, column=0, columnspan=1, sticky="w")
        parada_en_media_check = tk.Checkbutton(
            acciones_media_frame,
            text="Parar la ejecución",
            variable=self.parada_en_media,
            command=self.toggle_medias_entry,
        )
        parada_en_media_check.grid(row=1, column=0, sticky="w")

        self.max_medias_label = tk.Label(
            acciones_media_frame, text="Cantidad de medias iguales seguidas:"
        )
        self.max_medias_label.grid(row=0, column=0)

        self.max_medias_entry = tk.Entry(
            acciones_media_frame, textvariable=self.max_medias_iguales
        )
        self.max_medias_entry.grid(row=0, column=1)

        # Cambio de mutación
        cambio_de_mutacion_check = tk.Checkbutton(
            acciones_media_frame,
            text="Poner probabilidad de mutación a 0.5",
            variable=self.cambio_de_mutacion,
            command=self.toggle_medias_entry,
        )
        cambio_de_mutacion_check.grid(row=2, column=0, columnspan=1, sticky="w")

        # Parada en clones
        parada_en_clones_check = tk.Checkbutton(
            self.sin_biblioteca_frame,
            text="Parar si toda la población son el mismo individuo",
            variable=self.parada_en_clones,
        )
        parada_en_clones_check.grid(row=2, column=0, columnspan=1, sticky="w")

        # Dibujar resultados parciales
        plot_resultados_parciales_check = tk.Checkbutton(
            self.sin_biblioteca_frame,
            text="Mostrar mejor individuo en cada generación",
            variable=self.plot_resultados_parciales,
        )
        plot_resultados_parciales_check.grid(row=4, column=0, sticky="w")

    def toggle_medias_entry(self):
        if self.parada_en_media.get() or self.cambio_de_mutacion.get():
            self.max_medias_entry.configure(state="normal")
            self.max_medias_label.configure(state="normal")
        else:
            self.max_medias_entry.configure(state="disabled")
            self.max_medias_label.configure(state="disabled")

    def toggle_elitismo(self):
        state = "normal" if self.elitismo_var.get() else "disabled"
        for widget in self.elitismo_widgets:
            widget.configure(state=state)
            if state == "normal" and widget is self.num_padres_entry:
                if self.num_padres_pasados_activo:
                    widget.configure(state="normal")
                else:
                    widget.configure(state="disabled")

    def toggle_biblioteca(self):
        if self.usar_biblioteca.get():
            change_state_container(self.sin_biblioteca_frame, "disabled")

            self.mutacion_dropdown["menu"].delete(2)
            # Eliminamos los crossover que no estén implementados
            self.crossover_dropdown["menu"].delete(4)
            self.crossover_dropdown["menu"].delete(3)
            self.crossover_dropdown["menu"].delete(2)

            # Activamos el chebox de elitismo
            self.elitismo_var.set(True)
            self.toggle_elitismo()

            # Inhabilitamos el checkbox de elitismo
            self.elitismo_check.configure(state="disabled")

            # Si está seleccionado alguno de los que no hay, lo cambiamos a uno que sí
            if (
                self.crossover_tipo.get()
                == self._enums_to_string(Crossover.CROSSOVER_CYCLE)
                or self.crossover_tipo.get()
                == self._enums_to_string(Crossover.EDGE_RECOMBINATION_CROSSOVER)
                or self.crossover_tipo.get()
                == self._enums_to_string(Crossover.CROSSOVER_PDF)
            ):
                self.crossover_tipo.set(
                    self._enums_to_string(Crossover.CROSSOVER_ORDER)
                )

            if self.mutacion_tipo.get() == self._enums_to_string(
                Mutacion.INTERCAMBIAR_GENES_VECINOS
            ):
                self.mutacion_tipo.set(self._enums_to_string(Mutacion.PERMUTAR_ZONA))

        else:
            change_state_container(self.sin_biblioteca_frame, "normal")

            if self.num_padres_pasados_activo:
                self.num_padres_entry.configure(state="normal")
            # Insertamos la mutacion en el desplegable
            self.mutacion_dropdown["menu"].insert_command(
                2,
                label=self._enums_to_string(Mutacion.INTERCAMBIAR_GENES_VECINOS),
                command=lambda: self.mutacion_tipo.set(
                    self._enums_to_string(Mutacion.INTERCAMBIAR_GENES_VECINOS)
                ),
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

            # Habilitamos el check de elitismo
            self.elitismo_check.configure(state="normal")

    def pick_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        self.fichero_coordenadas.set(filename)

    def toggle_seleccion(self, value):
        if value != self._enums_to_string(Seleccion.TORNEO):
            self.participantes_torneo_label.configure(state="disabled")
            self.participantes_torneo_entry.configure(state="disabled")
        else:
            self.participantes_torneo_label.configure(state="normal")
            self.participantes_torneo_entry.configure(state="normal")

    def ejecutar(self):
        self.ejecutar_button.configure(state="disabled", text="Ejecutando...")

        # Create a worker thread to execute the code
        worker_thread = threading.Thread(target=self._execute_code)
        worker_thread.start()
        self.hilos.append(worker_thread)

    def _execute_code(self):
        try:
            if self.fichero_coordenadas.get() == "Ningún archivo seleccionado":
                raise tk.TclError("No coordinates file selected. Please select a file.")

            tipo_seleccion = Seleccion[
                self.seleccion_tipo.get().upper().replace(" ", "_")
            ]
            tipo_mutacion = Mutacion[self.mutacion_tipo.get().upper().replace(" ", "_")]
            tipo_crossover = Crossover[
                self.crossover_tipo.get().upper().replace(" ", "_")
            ]
            tipo_elitismo = Elitismo[self.elitismo_tipo.get().upper().replace(" ", "_")]

            if self.usar_biblioteca.get():
                deapTSP.ejecucion_paralela(
                    self.num_ejecuciones.get(),
                    self.fichero_coordenadas.get(),
                    self.verbose_var.get(),
                    self.num_iteraciones.get(),
                    self.prob_mutacion.get(),
                    self.prob_crossover.get(),
                    self.participantes_torneo.get(),
                    self.num_individuos.get(),
                    tipo_seleccion,
                    tipo_mutacion,
                    tipo_crossover,
                    tipo_elitismo,
                    self.num_padres_pasados.get(),
                )

            else:
                miTSP.ejecucion_paralela(
                    self.num_ejecuciones.get(),
                    self.fichero_coordenadas.get(),
                    self.dibujar_evolucion.get(),
                    self.verbose_var.get(),
                    self.parada_en_media.get(),
                    self.max_medias_iguales.get(),
                    self.elitismo_var.get(),
                    self.parada_en_clones.get(),
                    self.plot_resultados_parciales.get(),
                    self.cambio_de_mutacion.get(),
                    self.num_iteraciones.get(),
                    self.prob_mutacion.get(),
                    self.prob_crossover.get(),
                    self.participantes_torneo.get(),
                    self.num_individuos.get(),
                    tipo_seleccion,
                    tipo_mutacion,
                    tipo_crossover,
                    tipo_elitismo,
                    self.num_padres_pasados.get(),
                )

        except tk.TclError as e:
            # Lanzar un modal con el error:
            tk.messagebox.showerror("Error", str(e))

        finally:
            # Notify the main window that the work is done
            self.ejecutar_button.configure(state="normal", text="Ejecutar")

    def seleccion_elitismo(self, value):
        # Si se selecciona elitismo N Padres, se debe activar el número de padres a pasar. Hacer el mapeo con la enumeración
        if value == self._enums_to_string(Elitismo.PASAR_N_PADRES):
            self.num_padres_entry.configure(state="normal")
            self.num_padres_pasados_activo = True
        else:
            self.num_padres_entry.configure(state="disabled")
            self.num_padres_pasados_activo = False


def change_state_container(container, state):
    for child in container.winfo_children():
        change_state_container(child, state)
        try:
            child.configure(state=state)
        except tk.TclError:
            # El widget no tiene estado
            pass


# Inicializar la interfaz
if __name__ == "__main__":
    app = GeneticAlgorithmUI()
    app.mainloop()
