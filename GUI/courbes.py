
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from Model.rasnet34 import ResNet
from utils.data_loader import get_data_loaders


class TrainingDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image Classification Training")
        self.root.geometry("1200x800")

        # Training control variables
        self.is_training = False
        self.should_pause = False
        self.current_epoch = 0
        self.train_losses = []
        self.val_accuracies = []

        self.setup_ui()

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
        self.epochs_entry.insert(0, "10")
        self.epochs_entry.grid(row=0, column=1, sticky="ew", pady=2)

        ttk.Label(config_frame, text="Batch Size:").grid(row=1, column=0, sticky="w")
        self.batch_entry = ttk.Entry(config_frame)
        self.batch_entry.insert(0, "32")
        self.batch_entry.grid(row=1, column=1, sticky="ew", pady=2)

        ttk.Label(config_frame, text="Learning Rate:").grid(row=2, column=0, sticky="w")
        self.lr_entry = ttk.Entry(config_frame)
        self.lr_entry.insert(0, "0.001")
        self.lr_entry.grid(row=2, column=1, sticky="ew", pady=2)

        ttk.Label(config_frame, text="Number of Classes:").grid(row=3, column=0, sticky="w")
        self.classes_entry = ttk.Entry(config_frame)
        self.classes_entry.insert(0, "14")
        self.classes_entry.grid(row=3, column=1, sticky="ew", pady=2)

        ttk.Label(config_frame, text="Device:").grid(row=4, column=0, sticky="w")
        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        ttk.Radiobutton(config_frame, text="CPU", variable=self.device_var, value="cpu").grid(row=4, column=1,
                                                                                              sticky="w")
        ttk.Radiobutton(config_frame, text="GPU", variable=self.device_var, value="cuda").grid(row=5, column=1,
                                                                                               sticky="w")

        # Control buttons
        self.start_btn = ttk.Button(config_frame, text="Start Training", command=self.start_training)
        self.start_btn.grid(row=6, column=0, columnspan=2, pady=10, sticky="ew")

        self.pause_btn = ttk.Button(config_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_btn.grid(row=7, column=0, columnspan=2, pady=5, sticky="ew")

        self.stop_btn = ttk.Button(config_frame, text="Stop", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.grid(row=8, column=0, columnspan=2, pady=5, sticky="ew")

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

        # Matplotlib figure for live plotting
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure grid weights for resizing
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)

    def start_training(self):
        if self.is_training:
            return

        try:
            self.num_epochs = int(self.epochs_entry.get())
            self.batch_size = int(self.batch_entry.get())
            self.lr = float(self.lr_entry.get())
            self.num_classes = int(self.classes_entry.get())
            self.device = torch.device(self.device_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid input values")
            return

        # Disable controls during training
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)

        # Reset UI components
        self.progress["value"] = 0
        self.metrics_text.delete(1.0, tk.END)
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()
        self.train_losses = []
        self.val_accuracies = []

        # Initialize training state variables
        self.is_training = True
        self.should_pause = False
        self.current_epoch = 0

        # Start training thread
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
        # Initialize model, criterion and optimizer
        model = ResNet(self.num_classes).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Get data loaders
        train_loader, val_loader, _ = get_data_loaders(batch_size=self.batch_size)

        for epoch in range(self.current_epoch, self.num_epochs):
            if not self.is_training:
                break

            # Pause handling
            while self.should_pause:
                if not self.is_training:
                    return
                time.sleep(0.1)

            # Training phase
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            # Calculate epoch metrics
            epoch_loss = running_loss / len(train_loader.dataset)
            self.train_losses.append(epoch_loss)

            # Validation phase
            val_accuracy = self.evaluate(model, val_loader)
            self.val_accuracies.append(val_accuracy)

            # Update UI
            self.current_epoch = epoch + 1
            progress = (epoch + 1) / self.num_epochs * 100
            self.progress["value"] = progress

            metrics_str = f"Epoch [{epoch + 1}/{self.num_epochs}]\n"
            metrics_str += f"Train Loss: {epoch_loss:.4f}\n"
            metrics_str += f"Validation Accuracy: {val_accuracy:.2f}%\n\n"

            self.metrics_text.insert(tk.END, metrics_str)
            self.metrics_text.see(tk.END)

            # Update plots
            self.update_plots()

            # Small delay to allow UI updates
            time.sleep(0.1)

        # Training complete
        self.is_training = False
        self.root.after(0, self.training_complete)

        # Save model
        torch.save(model.state_dict(), "saved_models/chestmnist_resnet34.pth")

    def evaluate(self, model, data_loader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                outputs = model(images)
                predicted = torch.sigmoid(outputs) > 0.5
                correct += (predicted == labels.bool()).sum().item()
                total += labels.numel()

        accuracy = 100 * correct / total
        return accuracy

    def update_plots(self):
        self.ax1.clear()
        self.ax2.clear()

        # Plot training loss
        self.ax1.plot(range(1, len(self.train_losses) + 1), self.train_losses, 'b-', marker='o')
        self.ax1.set_title('Training Loss')
        self.ax1.set_ylabel('Loss')

        # Plot validation accuracy
        self.ax2.plot(range(1, len(self.val_accuracies) + 1), self.val_accuracies, 'r-', marker='o')
        self.ax2.set_title('Validation Accuracy')
        self.ax2.set_ylabel('Accuracy (%)')
        self.ax2.set_xlabel('Epoch')

        self.canvas.draw()

    def training_complete(self):
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        messagebox.showinfo("Training Complete", "Model training has finished!")


if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingDashboard(root)
    root.mainloop()