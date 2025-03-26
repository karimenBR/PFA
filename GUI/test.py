import tkinter as tk
from datetime import time
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import torch
from Model.resnet18_model import ResNet
from utils.data_loader import get_data_loaders
from utils.visualize import visualize_predictions


class TrainingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PyTorch Training Dashboard")
        self.root.geometry("1200x800")
        self.setup_ui()

        # Training control variables
        self.is_training = False
        self.should_pause = False
        self.current_epoch = 0

    def setup_ui(self):
        # Configure styles
        style = ttk.Style()
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 10))
        style.configure('TButton', font=('Helvetica', 10))
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Configuration
        config_frame = ttk.LabelFrame(main_frame, text="Training Configuration", padding=10)
        config_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Configuration widgets
        ttk.Label(config_frame, text="Epochs:").grid(row=0, column=0, sticky="w")
        self.epochs_entry = ttk.Entry(config_frame)
        self.epochs_entry.insert(0, "100")
        self.epochs_entry.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(config_frame, text="Batch Size:").grid(row=1, column=0, sticky="w")
        self.batch_entry = ttk.Entry(config_frame)
        self.batch_entry.insert(0, "128")
        self.batch_entry.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(config_frame, text="Learning Rate:").grid(row=2, column=0, sticky="w")
        self.lr_entry = ttk.Entry(config_frame)
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(config_frame, text="Device:").grid(row=3, column=0, sticky="w")
        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        ttk.Radiobutton(config_frame, text="CPU", variable=self.device_var, value="cpu").grid(row=3, column=1,
                                                                                              sticky="w")
        ttk.Radiobutton(config_frame, text="GPU", variable=self.device_var, value="cuda").grid(row=4, column=1,
                                                                                               sticky="w")

        # Control buttons
        self.start_btn = ttk.Button(config_frame, text="Start Training", command=self.start_training)
        self.start_btn.grid(row=5, column=0, columnspan=2, pady=10, sticky="ew")

        self.pause_btn = ttk.Button(config_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_btn.grid(row=6, column=0, columnspan=2, pady=5, sticky="ew")

        self.stop_btn = ttk.Button(config_frame, text="Stop", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.grid(row=7, column=0, columnspan=2, pady=5, sticky="ew")

        # Right panel - Visualizations
        viz_frame = ttk.Frame(main_frame)
        viz_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=5, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(viz_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(fill=tk.X, pady=10)

        # Metrics display
        metrics_frame = ttk.LabelFrame(viz_frame, text="Training Metrics", padding=10)
        metrics_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.metrics_text = tk.Text(metrics_frame, height=10, width=50)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)

        # Matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)

    def start_training(self):
        if self.is_training:
            return

        try:
            self.epochs = int(self.epochs_entry.get())
            self.batch_size = int(self.batch_entry.get())
            self.lr = float(self.lr_entry.get())
            self.device = self.device_var.get()
        except ValueError:
            messagebox.showerror("Error", "Invalid input values")
            return

        # Disable controls during training
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)

        # Reset UI
        self.progress["value"] = 0
        self.metrics_text.delete(1.0, tk.END)
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()

        # Start training thread
        self.is_training = True
        self.should_pause = False
        self.current_epoch = 0

        training_thread = threading.Thread(target=self.run_training)
        training_thread.daemon = True
        training_thread.start()

    def toggle_pause(self):
        self.should_pause = not self.should_pause
        self.pause_btn.config(text="Resume" if self.should_pause else "Pause")

    def stop_training(self):
        self.is_training = False
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)

    def run_training(self):
        # Initialize training (simplified - replace with your actual training code)
        device = torch.device(self.device)
        model = ResNet(num_classes=14).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        train_loader, val_loader, _ = get_data_loaders(self.batch_size)

        # Training loop
        for epoch in range(self.current_epoch, self.epochs):
            if not self.is_training:
                break

            while self.should_pause:
                if not self.is_training:
                    return
                self.root.update()
                continue

            # Update UI
            self.current_epoch = epoch
            progress = (epoch + 1) / self.epochs * 100
            self.progress["value"] = progress
            self.root.title(f"Training - Epoch {epoch + 1}/{self.epochs}")

            # Simulate training (replace with actual training)
            train_loss = 0.8 * (1 - epoch / self.epochs)  # Simulated loss
            val_acc = 0.6 + 0.3 * (epoch / self.epochs)  # Simulated accuracy

            # Update metrics
            self.metrics_text.insert(tk.END, f"Epoch {epoch + 1}: Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%\n")
            self.metrics_text.see(tk.END)

            # Update plots
            self.ax1.plot(epoch, train_loss, 'bo-', label='Training Loss')
            self.ax2.plot(epoch, val_acc, 'ro-', label='Validation Accuracy')
            self.ax1.set_ylabel("Loss")
            self.ax2.set_ylabel("Accuracy")
            self.ax2.set_xlabel("Epoch")
            self.ax1.legend()
            self.ax2.legend()
            self.canvas.draw()

            # Simulate batch processing time
            time.sleep(0.1)

        # Training complete
        self.is_training = False
        self.root.after(0, self.training_complete)

    def training_complete(self):
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.root.title("PyTorch Training Dashboard")
        messagebox.showinfo("Info", "Training completed!")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingApp(root)
    root.mainloop()