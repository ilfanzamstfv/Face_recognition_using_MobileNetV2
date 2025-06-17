import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import json
from datetime import datetime
import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import face_recognition

# Load model dan class indices
MODEL_PATH = 'model/face_classifier_model.h5' # sesuaikan dengan penyimpanan file
CLASS_INDICES_PATH = 'model/face_labels.json' # sesuaikan dengan penyimpanan file
TRANSACTION_HISTORY_FILE = 'transaction_history.csv' # sesuaikan dengan penyimpanan file
FACE_DETECTION_MODEL = "hog"
USERS_DB_PATH = 'users_db.json' # sesuaikan dengan penyimpanan file
CONFIDENCE_THRESHOLD = 0.85

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Database pengguna (simulasi)
try:
    with open(USERS_DB_PATH, 'r') as f:
        users_db = json.load(f)
    print(f"Loaded {len(users_db)} users from database")
except FileNotFoundError:
    print("User database not found. Creating a new one...")
    users_db = {
        "dzakwan": {"password": "dzakwan123", "name": "Dzakwan", "balance": 3000000, "face_class": "dzakwan"},
        "fanza": {"password": "fanza123", "name": "Fanza", "balance": 1000000, "face_class": "fanza"},
        "fillo": {"password": "fillo123", "name": "Fillo", "balance": 500000, "face_class": "fillo"},
        "yeru": {"password": "yeru123", "name": "Yeru", "balance": 500000, "face_class": "yeru"},
        "zahid": {"password": "zahid123", "name": "Zahid", "balance": 1000000, "face_class": "zahid"},
    }

    with open(USERS_DB_PATH, 'w') as f:
        json.dump(users_db, f, indent=2)
    print("Created default user database")
except json.JSONDecodeError:
    print("Error in user database format. Using default database...")
    users_db = {
        "dzakwan": {"password": "dzakwan123", "name": "Dzakwan", "balance": 3000000, "face_class": "dzakwan"},
        "fanza": {"password": "fanza123", "name": "Fanza", "balance": 1000000, "face_class": "fanza"},
        "fillo": {"password": "fillo123", "name": "Fillo", "balance": 500000, "face_class": "fillo"},
        "yeru": {"password": "yeru123", "name": "Yeru", "balance": 500000, "face_class": "yeru"},
        "zahid": {"password": "zahid123", "name": "Zahid", "balance": 1000000, "face_class": "zahid"},
    }
    
    with open(USERS_DB_PATH, 'w') as f:
        json.dump(users_db, f, indent=2)
    print("Created default user database")

def save_users_db():
    try:
        with open(USERS_DB_PATH, 'w') as f:
            json.dump(users_db, f, indent=2)
        print("User database saved successfully")
    except Exception as e:
        print(f"Error saving user database: {e}")

# Database untuk riwayat transaksi
if not os.path.exists(TRANSACTION_HISTORY_FILE):
    with open(TRANSACTION_HISTORY_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Sender', 'Receiver', 'Amount', 'Type', 'Status'])

class FacePaymentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Paysecure Digital Payment")
        self.root.geometry("1000x700")
        
        self.current_user = None
        self.camera_active = False
        self.cap = None
        
        # Style configuration
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10))
        self.style.configure('Header.TLabel', font=('Arial', 16, 'bold'))
        self.style.configure('Red.TLabel', foreground='red')
        self.style.configure('Green.TLabel', foreground='green')
        
        self.create_login_screen()
    
    def create_login_screen(self):
        """Create the login screen"""
        self.clear_window()
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, padx=50, pady=50)
        
        ttk.Label(main_frame, text="Pay Secure Login", style='Header.TLabel').grid(row=0, column=0, columnspan=2, pady=20)
        
        ttk.Label(main_frame, text="Username:").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.username_entry = ttk.Entry(main_frame)
        self.username_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(main_frame, text="Password:").grid(row=2, column=0, sticky='e', padx=5, pady=5)
        self.password_entry = ttk.Entry(main_frame, show="*")
        self.password_entry.grid(row=2, column=1, padx=5, pady=5)
        
        login_button = ttk.Button(main_frame, text="Login", command=self.login)
        login_button.grid(row=3, column=0, columnspan=2, pady=20)
    
    def login(self):
        """Handle login process"""
        username = self.username_entry.get()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Error", "Username and password are required")
            return
        
        if username not in users_db or users_db[username]["password"] != password:
            messagebox.showerror("Error", "Invalid username or password")
            return
        
        self.current_user = username
        self.create_main_menu()
    
    def create_main_menu(self):
        """Create the main menu after login"""
        self.clear_window()
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, padx=20, pady=20)
        
        # Header
        user_info = users_db[self.current_user]
        header_text = f"Welcome, {user_info['name']} | Balance: Rp {user_info['balance']:,}"
        ttk.Label(main_frame, text=header_text, style='Header.TLabel').pack(pady=20)
        
        # Menu buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Send Money", command=self.send_money).grid(row=0, column=0, padx=10, pady=10)
        ttk.Button(button_frame, text="Transaction History", command=self.show_history).grid(row=0, column=1, padx=10, pady=10)
        ttk.Button(button_frame, text="Logout", command=self.logout).grid(row=1, column=0, columnspan=2, padx=10, pady=10)
    
    def send_money(self):
        """Send money to another user"""
        self.clear_window()
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, padx=20, pady=20)
        
        ttk.Label(main_frame, text="Send Money", style='Header.TLabel').pack(pady=20)
        
        # Recipient input
        ttk.Label(main_frame, text="Recipient Username:").pack(pady=5)
        self.recipient_entry = ttk.Entry(main_frame)
        self.recipient_entry.pack(pady=5)
        
        # Amount input
        ttk.Label(main_frame, text="Amount:").pack(pady=5)
        self.amount_entry = ttk.Entry(main_frame)
        self.amount_entry.pack(pady=5)
        
        # Send button
        ttk.Button(main_frame, text="Verify & Send", command=self.verify_and_send).pack(pady=20)
        
        # Back button
        ttk.Button(main_frame, text="Back", command=self.create_main_menu).pack(pady=10)
    
    def verify_and_send(self):
        """Verify face and send money"""
        recipient = self.recipient_entry.get()
        amount_str = self.amount_entry.get()
        
        if not recipient or not amount_str:
            messagebox.showerror("Error", "Recipient and amount are required")
            return
        
        try:
            amount = int(amount_str)
        except ValueError:
            messagebox.showerror("Error", "Invalid amount")
            return
        
        if recipient not in users_db:
            messagebox.showerror("Error", "Recipient not found")
            return
        
        if users_db[self.current_user]["balance"] < amount:
            messagebox.showerror("Error", "Insufficient balance")
            return
        
        # Verify face before sending
        verify_window = tk.Toplevel(self.root)
        verify_window.title("Face Verification")
        verify_window.geometry("600x500")
        
        ttk.Label(verify_window, text="Please verify your face to send money", style='Header.TLabel').pack(pady=10)
        
        # Frame untuk video
        video_frame = ttk.Frame(verify_window)
        video_frame.pack(pady=10)

        face_label = ttk.Label(verify_window)
        face_label.pack(pady=10)
        
        status_label = ttk.Label(verify_window, text="Verification in progress...")
        status_label.pack(pady=10)

        # Mulai kamera untuk verifikasi
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open camera")
            verify_window.destroy()
            return

        def update_frame():
            ret, frame = cap.read()
            if not ret:
                status_label.config(text="Camera error", style='Red.TLabel')
                return

            # Resize frame untuk proses lebih cepat
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all face locations in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame, model=FACE_DETECTION_MODEL)
            
            # If faces detected
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                # Lakukan prediksi
                predictions = model.predict(np.array(face_encodings))

                for i, pred in enumerate(predictions):
                    # Dapatkan label dan confidence score
                    pred_label = np.argmax(pred)
                    confidence = np.max(pred)

                    # Skala kembali lokasi wajah ke ukuran frame asli
                    top, right, bottom, left = face_locations[i]
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    if confidence >= CONFIDENCE_THRESHOLD:
                        predicted_name = class_names.get(pred_label, "Unknown")
                        display_color = (0, 255, 0)  # Hijau
                        style_label = 'Green.TLabel'

                    else:
                        predicted_name = "Unknown"
                        display_color = (0, 0, 255)  # Merah
                        style_label = 'Red.TLabel'

                    # # Gambar kotak dan label
                    # cv2.rectangle(frame, (left, top), (right, bottom), display_color, 2)
                    # label = f"{predicted_name} ({confidence:.2f})"
                    # cv2.putText(frame, label, (left, top-10), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, display_color, 2)
                    
                    # status_label.config(
                    #     text=f"Detected: {predicted_name} (Confidence: {confidence:.2f})",
                    #     style=style_label
                    # )

                    # If verification successful
                    if (confidence >= CONFIDENCE_THRESHOLD and 
                        users_db[self.current_user]["face_class"] == predicted_name.lower()):
                        status_label.config(text=f"Verified as {predicted_name}!", style='Green.TLabel')
                        cap.release()
                        verify_window.after(3000, lambda: self.finalize_send(verify_window, recipient, amount))
                        return

            # Display the frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            face_label.imgtk = imgtk
            face_label.configure(image=imgtk)

            verify_window.after(10, update_frame)

        # tombol cancel
        cancel_button = ttk.Button(verify_window, text="Cancel", command=lambda: [cap.release(), verify_window.destroy()])
        cancel_button.pack(pady=10)
        
        update_frame()
    
    def finalize_send(self, window, recipient, amount):
        """Finalize the money transfer"""
        window.destroy()
        
        # Perform the transfer
        users_db[self.current_user]["balance"] -= amount
        users_db[recipient]["balance"] += amount
        
        # Record transaction
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(TRANSACTION_HISTORY_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, self.current_user, recipient, amount, "Money Transfer", "Completed"])
        
        messagebox.showinfo("Success", f"Sent Rp {amount:,} to {recipient} successfully!")
        self.create_main_menu()
    
    def show_history(self):
        """Show transaction history"""
        self.clear_window()
        
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        
        ttk.Label(main_frame, text="Transaction History", style='Header.TLabel').pack(pady=20)
        
        # Create treeview
        columns = ('timestamp', 'sender', 'receiver', 'amount', 'type', 'status')
        tree = ttk.Treeview(main_frame, columns=columns, show='headings')
        
        # Define headings
        tree.heading('timestamp', text='Date & Time')
        tree.heading('sender', text='Sender')
        tree.heading('receiver', text='Receiver')
        tree.heading('amount', text='Amount')
        tree.heading('type', text='Type')
        tree.heading('status', text='Status')
        
        # Set column widths
        tree.column('timestamp', width=150)
        tree.column('sender', width=100)
        tree.column('receiver', width=100)
        tree.column('amount', width=100)
        tree.column('type', width=120)
        tree.column('status', width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        tree.pack(expand=True, fill='both')
        
        # Load data
        try:
            with open(TRANSACTION_HISTORY_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                for row in reader:
                    if row[1] == self.current_user or row[2] == self.current_user:
                        tree.insert('', 'end', values=row)
        except FileNotFoundError:
            messagebox.showerror("Error", "Transaction history not found")
        
        # Back button
        ttk.Button(main_frame, text="Back", command=self.create_main_menu).pack(pady=20)
    
    def logout(self):
        """Logout the current user"""
        self.current_user = None
        self.create_login_screen()
    
    def clear_window(self):
        """Clear all widgets from the window"""
        for widget in self.root.winfo_children():
            widget.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacePaymentApp(root)
    root.mainloop()
