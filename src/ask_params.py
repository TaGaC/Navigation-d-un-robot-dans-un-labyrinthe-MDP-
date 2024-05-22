import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

class ParamInputApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paramètres du programme")
        
        # Configure the grid to be responsive
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        for i in range(5):
            self.root.rowconfigure(i, weight=1)
        
        # Variables for gamma and grid size
        self.gamma = None
        self.grid_size = None
        
        # Gamma input
        self.gamma_label = tk.Label(root, text="Gamma")
        self.gamma_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.gamma_label_left = tk.Label(root, text="Ɣ (entre 0 et 1)")
        self.gamma_label_left.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.gamma_entry = tk.Entry(root)
        self.gamma_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Grid size input
        self.grid_size_label = tk.Label(root, text="Taille du quadrillage")
        self.grid_size_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        self.grid_size_entry = tk.Entry(root)
        self.grid_size_entry.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # Submit button
        self.submit_button = tk.Button(root, text="Soumettre", command=self.submit)
        self.submit_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # Bind the <Configure> event to adjust the font size and widget dimensions
        self.root.bind("<Configure>", self.adjust_widgets)

    def adjust_widgets(self, event):
        # Calculate new font size based on window height
        new_font_size = max(8, int(self.root.winfo_height() / 25))
        font = ("TkDefaultFont", new_font_size)

        # Update font for all widgets
        for widget in [self.gamma_label, self.gamma_label_left, self.gamma_entry, self.grid_size_label, self.grid_size_entry, self.submit_button]:
            widget.config(font=font)

        # Update padding for better visual adaptation
        padding = int(self.root.winfo_height() / 50)
        for widget in [self.gamma_label, self.gamma_label_left, self.gamma_entry, self.grid_size_label, self.grid_size_entry, self.submit_button]:
            widget.grid_configure(padx=padding, pady=padding)

    def submit(self):
        try:
            self.gamma = float(self.gamma_entry.get())
            if self.gamma < 0 or self.gamma > 1:
                raise ValueError("Gamma doit être entre 0 et 1")
            
            self.grid_size = int(self.grid_size_entry.get())
            if self.grid_size < 1 or self.grid_size > 10:
                raise ValueError("La taille du quadrillage doit être comprise entre 1 et 10")
            
            # Call a function to create and display the grid
            self.display_grid(self.grid_size)

        except ValueError as e:
            messagebox.showerror("Erreur", f"Entrée invalide: {e}")

    def display_grid(self, grid_size):
        self.grid_window = tk.Toplevel(self.root)
        self.grid_window.title("Quadrillage")

        # Déterminez la taille de l'écran
        screen_width = self.grid_window.winfo_screenwidth()
        screen_height = self.grid_window.winfo_screenheight()

        # Déterminez la taille de la fenêtre
        window_width = min(screen_width, screen_height) * 0.8  # Largeur de la fenêtre (80% de la plus petite dimension de l'écran)
        window_height = window_width  # Hauteur de la fenêtre (même que la largeur pour un carré)

        # Calculez les coordonnées x et y pour centrer la fenêtre
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2

        # Vérifiez que window_width et window_height ne sont pas nuls
        if window_width <= 0 or window_height <= 0:
            # Si l'une des dimensions est nulle ou négative, définissez une taille par défaut
            window_width = 400
            window_height = 400
            x = (screen_width - window_width) // 2
            y = (screen_height - window_height) // 2

        # Créez la géométrie de la fenêtre avec les coordonnées calculées

        self.grid_window.geometry(f"{int(window_width)}x{int(window_height)}+{int(x)}+{int(y)}")

        # Make the grid_window responsive
        for i in range(grid_size):
            self.grid_window.rowconfigure(i, weight=1)
            self.grid_window.columnconfigure(i, weight=1)

        # Variables to store start and end positions
        self.start_pos = None
        self.end_pos = None
        self.wall_state = False
        self.walls = []

        # Create and display the grid
        for i in range(grid_size):
            for j in range(grid_size):
                cell = tk.Label(self.grid_window, text="", width=5, height=2, relief="ridge")
                cell.grid(row=i, column=j, sticky="nsew")
                cell.bind("<Button-1>", lambda event, row=i, col=j: self.cell_clicked(event, row, col))

        # Add a button to finish wall selection, initially disabled
        self.finish_button = tk.Button(self.grid_window, text="Terminer la sélection des murs", state=tk.DISABLED, command=self.finish_walls)
        self.finish_button.grid(row=grid_size, columnspan=grid_size, sticky="nsew")

        # Make the button row responsive
        self.grid_window.rowconfigure(grid_size, weight=0.25)

        # Bind the <Configure> event to adjust the font size and widget dimensions in the grid window
        self.grid_window.bind("<Configure>", self.adjust_grid_widgets)

    def adjust_grid_widgets(self, event):
        grid_size = self.grid_size

        # Calculate new font size based on window height
        new_font_size = max(8, int(self.grid_window.winfo_height() / (grid_size * 2)))
        font = ("TkDefaultFont", new_font_size)

        # Update font for all cell widgets
        for i in range(grid_size):
            for j in range(grid_size):
                widget = self.grid_window.grid_slaves(row=i, column=j)[0]
                widget.config(font=font)

        # Update padding for better visual adaptation
        padding = int(self.grid_window.winfo_height() / (grid_size * 25))
        for i in range(grid_size):
            for j in range(grid_size):
                widget = self.grid_window.grid_slaves(row=i, column=j)[0]
                widget.grid_configure(padx=padding, pady=padding)

        # Update font and padding for the finish button
        self.finish_button.config(font=font)
        self.finish_button.grid_configure(padx=padding, pady=padding)

    def cell_clicked(self, event, row, col):
        if self.start_pos is None:
            self.start_pos = (row, col)
            event.widget.config(bg="green")
            print(f"Point de départ défini: ({row}, {col})")
        elif self.end_pos is None:
            self.end_pos = (row, col)
            event.widget.config(bg="red")
            print(f"Point d'arrivée défini: ({row}, {col})")
            # Enable the finish button once the start and end positions are defined
            self.finish_button.config(state=tk.NORMAL)
        else:
            # Add wall
            self.walls.append((row, col))
            event.widget.config(bg="black")
            print(f"Ajout d'un mur à la position: ({row}, {col})")

    def finish_walls(self):
        # Change wall state to True to stop adding walls
        self.wall_state = True
        print("Sélection des murs terminée.")
        
        # Close the grid window and the main window
        self.grid_window.destroy()
        self.root.destroy()

        # Print the collected parameters for verification
        print(f"Gamma: {self.gamma}")
        print(f"Taille du quadrillage: {self.grid_size}")
        print(f"Position de départ: {self.start_pos}")
        print(f"Position de fin: {self.end_pos}")
        print(f"Positions des murs: {self.walls}")

if __name__ == "__main__":
    root = tk.Tk()
    # Déterminez la taille de l'écran
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    # Déterminez la taille de la fenêtre
    window_width = 800  # Remplacez par la largeur de votre fenêtre
    window_height = 500  # Remplacez par la hauteur de votre fenêtre

    # Calculez les coordonnées x et y pour centrer la fenêtre
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Créez la géométrie de la fenêtre avec les coordonnées calculées
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    app = ParamInputApp(root)
    root.mainloop()
