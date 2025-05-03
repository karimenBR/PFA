import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import TkinterDnD, DND_FILES
import json
from datetime import datetime
import os
from PIL import Image, ImageTk
import torch
from Model.rasnet34 import ResNet
import torch.nn.functional as F
from torchvision import transforms


class MedicalAnalysisApp(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title("Medical Image Analysis System")
        self.geometry("1200x800")
        self.configure(bg='#f0f8ff')

        # Initialize model with error handling
        self.model = None
        self.initialize_model()

        # Create variables for form fields
        self.patient_data = {
            "first_name": tk.StringVar(),
            "last_name": tk.StringVar(),
            "CIN": tk.StringVar(),
            "dob": tk.StringVar(),
            "gender": tk.StringVar(value="Male"),
            "image_path": "",
            "prediction": "",
            "confidence": "",
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        self.setup_ui()
        self.create_json_storage()

    def initialize_model(self):
        try:
            model_path = os.path.join("saved_models", "chestmnist_resnet34.pth")

            # Create directory if it doesn't exist
            os.makedirs("saved_models", exist_ok=True)

            if not os.path.exists(model_path):
                # Try to find the model file in parent directories
                for root, dirs, files in os.walk(".."):
                    if "chestmnist_resnet34.pth" in files:
                        model_path = os.path.join(root, "chestmnist_resnet34.pth")
                        break
                else:
                    raise FileNotFoundError("Model file not found in any parent directory")

            self.model = ResNet(num_classes=14)
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.model.eval()

        except Exception as e:
            messagebox.showerror("Model Loading Error",
                                 f"Failed to load model: {str(e)}\n\n"
                                 "Please ensure:\n"
                                 "1. The model file 'chestmnist_resnet34.pth' exists\n"
                                 "2. It's placed in the 'saved_models' folder\n"
                                 "3. The file is not corrupted")
            self.destroy()
            raise

    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left panel - Patient Information
        info_frame = ttk.LabelFrame(main_frame, text="Patient Information", padding=15)
        info_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Form fields
        ttk.Label(info_frame, text="First Name:").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(info_frame, textvariable=self.patient_data["first_name"], width=30).grid(row=0, column=1, pady=5)

        ttk.Label(info_frame, text="Last Name:").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(info_frame, textvariable=self.patient_data["last_name"], width=30).grid(row=1, column=1, pady=5)

        ttk.Label(info_frame, text="CIN:").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(info_frame, textvariable=self.patient_data["CIN"], width=30).grid(row=2, column=1, pady=5)

        ttk.Label(info_frame, text="Date of Birth (YYYY-MM-DD):").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(info_frame, textvariable=self.patient_data["dob"], width=30).grid(row=3, column=1, pady=5)

        ttk.Label(info_frame, text="Gender:").grid(row=4, column=0, sticky="w", pady=5)
        gender_combo = ttk.Combobox(info_frame, textvariable=self.patient_data["gender"],
                                    values=["Male", "Female", "Other"], width=27, state="readonly")
        gender_combo.grid(row=4, column=1, pady=5)

        # Right panel - Image Analysis
        analysis_frame = ttk.LabelFrame(main_frame, text="Image Analysis", padding=15)
        analysis_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Image display area - Using Canvas instead of Label for better control
        self.image_canvas = tk.Canvas(analysis_frame, width=400, height=300, bg='white', relief="solid")
        self.image_canvas.grid(row=0, column=0, pady=10, columnspan=2)
        self.image_canvas.create_text(200, 150, text="Drag & Drop Image Here\nor\nClick to Browse",
                                      fill="gray", font=('Helvetica', 12), justify="center")
        self.image_canvas.bind("<Button-1>", self.browse_image)
        self.image_canvas.drop_target_register(DND_FILES)
        self.image_canvas.dnd_bind('<<Drop>>', self.handle_drop)
        self.image_on_canvas = None

        # Prediction results
        results_frame = ttk.Frame(analysis_frame)
        results_frame.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

        ttk.Label(results_frame, text="Prediction:").grid(row=0, column=0, sticky="w")
        self.prediction_label = ttk.Label(results_frame, text="", font=('Helvetica', 10, 'bold'))
        self.prediction_label.grid(row=0, column=1, sticky="w")

        ttk.Label(results_frame, text="Confidence:").grid(row=1, column=0, sticky="w")
        self.confidence_label = ttk.Label(results_frame, text="", font=('Helvetica', 10, 'bold'))
        self.confidence_label.grid(row=1, column=1, sticky="w")

        # Analysis button
        analyze_btn = ttk.Button(analysis_frame, text="Analyze Image", command=self.analyze_image)
        analyze_btn.grid(row=2, column=0, pady=10, sticky="ew")

        save_btn = ttk.Button(analysis_frame, text="Save Record", command=self.save_to_json)
        save_btn.grid(row=2, column=1, pady=10, sticky="ew")

        # Bottom panel - Recent Analyses
        history_frame = ttk.LabelFrame(main_frame, text="Recent Analyses", padding=15)
        history_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

        self.history_tree = ttk.Treeview(history_frame, columns=("Name", "Date", "Prediction", "Confidence"),
                                         show="headings")
        self.history_tree.heading("Name", text="Patient Name")
        self.history_tree.heading("Date", text="Analysis Date")
        self.history_tree.heading("Prediction", text="Prediction")
        self.history_tree.heading("Confidence", text="Confidence")

        # Add scrollbar
        scrollbar = ttk.Scrollbar(history_frame, orient="vertical", command=self.history_tree.yview)
        scrollbar.pack(side="right", fill="y")
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        self.history_tree.pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)

        # Load recent analyses
        self.load_recent_analyses()

    def create_json_storage(self):
        self.data_file = "patient_records.json"
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                json.dump([], f)

    def browse_image(self, event=None):
        file_path = filedialog.askopenfilename(
            title="Select Medical Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.load_image(file_path)

    def handle_drop(self, event):
        file_path = event.data.strip("{}")  # Remove curly braces from path
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            self.load_image(file_path)
        else:
            messagebox.showerror("Error", "Please drop an image file (PNG, JPG, JPEG, BMP)")

    def load_image(self, file_path):
        try:
            self.patient_data["image_path"] = file_path
            img = Image.open(file_path)
            img.thumbnail((400, 300))  # Fit within our canvas size

            # Clear previous image
            self.image_canvas.delete("all")

            # Convert to PhotoImage and display
            photo = ImageTk.PhotoImage(img)
            self.image_on_canvas = photo  # Keep reference
            self.image_canvas.create_image(200, 150, image=photo, anchor="center")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            # Reset to default state
            self.image_canvas.delete("all")
            self.image_canvas.create_text(200, 150, text="Drag & Drop Image Here\nor\nClick to Browse",
                                          fill="gray", font=('Helvetica', 12), justify="center")

    def analyze_image(self):
        if not self.patient_data["image_path"]:
            messagebox.showwarning("Warning", "Please select an image first")
            return

        try:
            # Preprocess the image
            image = Image.open(self.patient_data["image_path"])
            image_tensor = self.preprocess_image(image)

            # Make prediction
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, preds = torch.max(probabilities, 1)
                confidence = confidence.item()
                predicted_class = preds.item()

            # Update UI with results
            class_names = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                           "Mass", "Nodule", "Pneumonia", "Pneumothorax",
                           "Consolidation", "Edema", "Emphysema", "Fibrosis",
                           "Pleural Thickening", "Hernia"]

            self.patient_data["prediction"] = class_names[predicted_class]
            self.patient_data["confidence"] = f"{confidence * 100:.2f}%"

            self.prediction_label.config(text=self.patient_data["prediction"])
            self.confidence_label.config(text=self.patient_data["confidence"])

        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def preprocess_image(self, image):
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')

        # Convert to tensor and add single channel dimension
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # For single channel
        ])

        return transform(image).unsqueeze(0)  # Add batch dimension

        return transform(image).unsqueeze(0)

    def save_to_json(self):
        # Validate required fields
        if not all([self.patient_data["first_name"].get(),
                    self.patient_data["last_name"].get(),
                    self.patient_data["dob"].get()]):
            messagebox.showwarning("Warning", "Please fill all patient information fields")
            return

        if not self.patient_data["prediction"]:
            messagebox.showwarning("Warning", "Please analyze an image first")
            return

        try:
            # Prepare record
            record = {
                "first_name": self.patient_data["first_name"].get(),
                "last_name": self.patient_data["last_name"].get(),
                "dob": self.patient_data["dob"].get(),
                "gender": self.patient_data["gender"].get(),
                "image_path": self.patient_data["image_path"],
                "prediction": self.patient_data["prediction"],
                "confidence": self.patient_data["confidence"],
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Load existing data
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            # Add new record
            data.append(record)

            # Save back to file
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=4)

            messagebox.showinfo("Success", "Patient record saved successfully")
            self.load_recent_analyses()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save record: {str(e)}")

    def load_recent_analyses(self):
        try:
            # Clear existing items
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)

            # Load data
            with open(self.data_file, 'r') as f:
                data = json.load(f)

            # Add to treeview (show last 10 records)
            for record in reversed(data[-10:]):
                name = f"{record['first_name']} {record['last_name']}"
                self.history_tree.insert("", "end",
                                         values=(name, record['analysis_date'],
                                                 record['prediction'], record['confidence']))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load recent analyses: {str(e)}")


if __name__ == "__main__":
    app = MedicalAnalysisApp()
    app.mainloop()