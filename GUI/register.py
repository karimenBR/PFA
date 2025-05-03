import tkinter as tk
from tkinter import ttk, messagebox
import hashlib
import json
from pathlib import Path


class RegisterWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Register New User")
        self.root.geometry("350x400")
        self.root.configure(bg='#f0f2f5')
        self.users_file = Path("users.json")

        self.setup_ui()

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def user_exists(self, full_name):
        if not self.users_file.exists():
            return False
        with open(self.users_file, 'r') as f:
            data = json.load(f)
            return any(user["full_name"] == full_name for user in data["users"])

    def register_user(self, full_name, password, fingerprint_ref):
        if not self.users_file.exists():
            data = {"users": []}
        else:
            with open(self.users_file, 'r') as f:
                data = json.load(f)

        data["users"].append({
            "full_name": full_name,
            "password": self.hash_password(password),
            "fingerprint": fingerprint_ref
        })

        with open(self.users_file, 'w') as f:
            json.dump(data, f, indent=2)

    def handle_register(self):
        full_name = self.name_entry.get().strip()
        password = self.pass_entry.get()
        confirm = self.confirm_entry.get()
        fingerprint = self.fingerprint_entry.get().strip()

        if not full_name or not password or not confirm or not fingerprint:
            messagebox.showwarning("Input Error", "Please fill all fields.")
            return

        if password != confirm:
            messagebox.showerror("Error", "Passwords do not match.")
            return

        if self.user_exists(full_name):
            messagebox.showerror("Error", "User already exists.")
            return

        self.register_user(full_name, password, fingerprint)
        messagebox.showinfo("Success", "User registered successfully!")
        self.root.destroy()

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg='#f0f2f5')
        main_frame.pack(pady=40, padx=30)

        tk.Label(main_frame, text="Register", font=('Helvetica', 16, 'bold'), bg='#f0f2f5', fg='#2c3e50').pack(pady=(0, 20))
        tk.Label(main_frame, text="New Assistant", font=('Helvetica', 12), bg='#f0f2f5', fg='#7f8c8d').pack(pady=(0, 20))

        tk.Label(main_frame, text="Full Name:", bg='#f0f2f5').pack(anchor='w')
        self.name_entry = ttk.Entry(main_frame, width=30)
        self.name_entry.pack(pady=5)

        tk.Label(main_frame, text="Password:", bg='#f0f2f5').pack(anchor='w')
        self.pass_entry = ttk.Entry(main_frame, width=30, show="\u2022")
        self.pass_entry.pack(pady=5)

        tk.Label(main_frame, text="Confirm Password:", bg='#f0f2f5').pack(anchor='w')
        self.confirm_entry = ttk.Entry(main_frame, width=30, show="\u2022")
        self.confirm_entry.pack(pady=5)

        tk.Label(main_frame, text="Fingerprint Reference:", bg='#f0f2f5').pack(anchor='w')
        self.fingerprint_entry = ttk.Entry(main_frame, width=30)
        self.fingerprint_entry.pack(pady=5)

        ttk.Button(main_frame, text="Register", command=self.handle_register).pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = RegisterWindow(root)
    root.mainloop()
