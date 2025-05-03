import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import json
import hashlib
from pathlib import Path


class SimplePFALogin:
    def __init__(self, root):
        self.root = root
        self.root.title("Login")
        self.root.geometry("350x400")
        self.root.resizable(False, False)
        self.root.configure(bg='#f0f2f5')

        # Path to users data JSON file
        self.users_file = Path("users.json")

        # Initialize if file doesn't exist
        if not self.users_file.exists():
            self.initialize_users_data()

        self.setup_ui()

    def initialize_users_data(self):
        """Create empty users data file"""
        default_data = {"users": []}
        with open(self.users_file, 'w') as f:
            json.dump(default_data, f, indent=2)

    def run_main_application(self):
        """Execute the main application file"""
        try:
            self.root.destroy()
            subprocess.Popen(["python", "DesktopInterface.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start application: {str(e)}")

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_user(self, full_name, password):
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)
            for user in data["users"]:
                if user["full_name"] == full_name:
                    return user["password"] == self.hash_password(password)
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read users data: {str(e)}")
            return False

    def register_user(self, full_name, password, fingerprint_ref):
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)

            for user in data["users"]:
                if user["full_name"] == full_name:
                    return False

            data["users"].append({
                "full_name": full_name,
                "password": self.hash_password(password),
                "fingerprint": fingerprint_ref
            })

            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to register user: {str(e)}")
            return False

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg='#f0f2f5')
        main_frame.pack(pady=40, padx=20, fill='both', expand=True)

        tk.Label(main_frame, text="Login", font=('Helvetica', 16, 'bold'), bg='#f0f2f5', fg='#2c3e50').pack(pady=(0, 20))
        tk.Label(main_frame, text="Assistant Login", font=('Helvetica', 12), bg='#f0f2f5', fg='#7f8c8d').pack(pady=(0, 30))

        form_frame = tk.Frame(main_frame, bg='#f0f2f5')
        form_frame.pack()

        tk.Label(form_frame, text="Full Name:", font=('Helvetica', 10), bg='#f0f2f5').grid(row=0, column=0, pady=5, sticky='w')
        self.name_entry = ttk.Entry(form_frame, width=25)
        self.name_entry.grid(row=1, column=0, pady=5)

        tk.Label(form_frame, text="Password:", font=('Helvetica', 10), bg='#f0f2f5').grid(row=2, column=0, pady=5, sticky='w')
        self.pass_entry = ttk.Entry(form_frame, width=25, show="\u2022")
        self.pass_entry.grid(row=3, column=0, pady=5)

        ttk.Button(form_frame, text="Login", command=self.handle_login, width=20).grid(row=4, column=0, pady=20)
        ttk.Button(form_frame, text="Register New User", command=self.show_register, style='secondary.TButton').grid(row=5, column=0)

        style = ttk.Style()
        style.configure('secondary.TButton', foreground='#3498db')

    def handle_login(self):
        full_name = self.name_entry.get().strip()
        password = self.pass_entry.get()

        if not full_name or not password:
            messagebox.showwarning("Input Error", "Please enter both full name and password")
            return

        if self.verify_user(full_name, password):
            messagebox.showinfo("Success", f"Welcome, {full_name}!")
            self.run_main_application()
        else:
            messagebox.showerror("Login Failed", "Invalid name or password")

    def show_register(self):
        subprocess.Popen(["python", "register.py"])


if __name__ == "__main__":
    root = tk.Tk()
    app = SimplePFALogin(root)
    root.mainloop()
