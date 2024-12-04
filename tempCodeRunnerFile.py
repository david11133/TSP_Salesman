def create_widgets(self):
        # Setup GUI components here (frames, labels, buttons, etc.)
        self.control_frame = tk.Frame(self, bg="#2C3E50")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            self.canvas_frame, bg="#ECF0F1", scrollregion=(0, 0, 1000, 1000)
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        
        self.canvas_scrollbar = ttk.Scrollbar(
            self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        self.canvas_scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.config(yscrollcommand=self.canvas_scrollbar.set)

        self.canvas_hscrollbar = ttk.Scrollbar(
            self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        self.canvas_hscrollbar.grid(row=1, column=0, sticky="ew")
        self.canvas.config(xscrollcommand=self.canvas_hscrollbar.set)

        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.create_stats_labels()
        self.create_buttons()
        self.create_input_fields()