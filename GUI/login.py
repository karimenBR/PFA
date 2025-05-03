import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
import json
import hashlib
import os
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
        default_data = {"users": {}}
        with open(self.users_file, 'w') as f:
            json.dump(default_data, f, indent=2)

    def run_main_application(self):
        """Execute the main application file"""
        try:
            # Close the login window
            self.root.destroy()

            # Run the main application file
            main_app_file = "DesktopInterface.py"  # Change this to your actual file name
            subprocess.Popen(["python", main_app_file])

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start application: {str(e)}")

    def hash_password(self, password):
        """Simple password hashing"""
        return hashlib.sha256(password.encode()).hexdigest()

    def verify_user(self, full_name, password):
        """Check if user exists and password matches"""
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)

            if full_name in data["users"]:
                stored_hash = data["users"][full_name]
                return stored_hash == self.hash_password(password)
            return False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read users data: {str(e)}")
            return False

    def register_user(self, full_name, password):
        """Add new user to JSON file"""
        try:
            with open(self.users_file, 'r') as f:
                data = json.load(f)

            if full_name in data["users"]:
                return False  # User already exists

            data["users"][full_name] = self.hash_password(password)

            with open(self.users_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Failed to register user: {str(e)}")
            return False

    def setup_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f2f5')
        main_frame.pack(pady=40, padx=20, fill='both', expand=True)

        # Logo/Title
        title = tk.Label(
            main_frame,
            text="Login",
            font=('Helvetica', 16, 'bold'),
            bg='#f0f2f5',
            fg='#2c3e50'
        )
        title.pack(pady=(0, 20))

        subtitle = tk.Label(
            main_frame,
            text="Assistant Login",
            font=('Helvetica', 12),
            bg='#f0f2f5',
            fg='#7f8c8d'
        )
        subtitle.pack(pady=(0, 30))

        # Login Form
        form_frame = tk.Frame(main_frame, bg='#f0f2f5')
        form_frame.pack()

        # Full Name
        tk.Label(
            form_frame,
            text="Full Name:",
            font=('Helvetica', 10),
            bg='#f0f2f5'
        ).grid(row=0, column=0, pady=5, sticky='w')

        self.name_entry = ttk.Entry(form_frame, width=25)
        self.name_entry.grid(row=1, column=0, pady=5)

        # Password
        tk.Label(
            form_frame,
            text="Password:",
            font=('Helvetica', 10),
            bg='#f0f2f5'
        ).grid(row=2, column=0, pady=5, sticky='w')

        self.pass_entry = ttk.Entry(form_frame, width=25, show="•")
        self.pass_entry.grid(row=3, column=0, pady=5)

        # Login Button
        login_btn = ttk.Button(
            form_frame,
            text="Login",
            command=self.handle_login,
            width=20
        )
        login_btn.grid(row=4, column=0, pady=20)

        # Register Button
        register_btn = ttk.Button(
            form_frame,
            text="Register New User",
            command=self.show_register,
            style='secondary.TButton'
        )
        register_btn.grid(row=5, column=0)

        # Configure styles
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
            # Here you would typically open the main application
            self.run_main_application()
        else:
            messagebox.showerror("Login Failed", "Invalid name or password")

    def show_register(self):
        """Show registration window"""
        register_win = tk.Toplevel(self.root)
        register_win.title("Register New User")
        register_win.geometry("300x300")
        register_win.resizable(False, False)
        register_win.configure(bg='#f0f2f5')

        # Registration Form
        tk.Label(
            register_win,
            text="Register New User",
            font=('Helvetica', 14, 'bold'),
            bg='#f0f2f5'
        ).pack(pady=20)

        # Full Name
        tk.Label(
            register_win,
            text="Full Name:",
            font=('Helvetica', 10),
            bg='#f0f2f5'
        ).pack(pady=(10, 0))

        reg_name_entry = ttk.Entry(register_win, width=25)
        reg_name_entry.pack(pady=5)

        # Password
        tk.Label(
            register_win,
            text="Password:",
            font=('Helvetica', 10),
            bg='#f0f2f5'
        ).pack(pady=(10, 0))

        reg_pass_entry = ttk.Entry(register_win, width=25, show="•")
        reg_pass_entry.pack(pady=5)

        # Register Button
        ttk.Button(
            register_win,
            text="Complete Registration",
            command=lambda: self.handle_register(
                reg_name_entry.get().strip(),
                reg_pass_entry.get(),
                register_win
            ),
            width=20
        ).pack(pady=20)

    def handle_register(self, full_name, password, window):
        if not full_name or not password:
            messagebox.showwarning("Input Error", "Please enter both full name and password")
            return

        if len(password) < 4:
            messagebox.showwarning("Weak Password", "Password should be at least 4 characters")
            return

        if self.register_user(full_name, password):
            messagebox.showinfo("Success", "Registration successful! You can now login.")
            window.destroy()
        else:
            messagebox.showerror("Error", "This name is already registered")


if __name__ == "__main__":
    root = tk.Tk()
    app = SimplePFALogin(root)
    root.mainloop()