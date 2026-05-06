import time
import threading
import json
import numpy as np
import customtkinter as ctk
import joblib
import os
import sys
from tkinter import messagebox

# Import local modules
sys.path.append(os.path.dirname(__file__))
from collect_keystrokes import extract_features, load_models
from enroll_and_predict import synthesize_samples, get_dataset_csv, append_to_dataset
from data_loader import prepare_data
from ensemble_voting import EnsembleVoter

# Design Palette
BG_DARK = "#0f172a"
SIDEBAR_COLOR = "#020617"
CARD_COLOR = "#1e293b"
ACCENT_BLUE = "#3b82f6"
SUCCESS_GREEN = "#22c55e"
ERROR_RED = "#ef4444"
TEXT_PRIMARY = "#f8fafc"
TEXT_SECONDARY = "#94a3b8"

USER_DB_PATH = "users.json"

def load_users():
    if not os.path.exists(USER_DB_PATH): return {}
    with open(USER_DB_PATH, 'r') as f: return json.load(f)

def save_users(users):
    with open(USER_DB_PATH, 'w') as f: json.dump(users, f, indent=4)

class ContinuousAuthMonitor:
    def __init__(self, callback_logout, callback_status, target_username):
        self.callback_logout = callback_logout
        self.callback_status = callback_status
        self.target_username = target_username
        self.current_events = []
        self.pressed_keys = set()
        self.voter = None
        self.scaler = None
        self.le = None
        self.is_monitoring = False
        
        try:
            self.voter, self.scaler, self.le, _ = load_models("models")
        except: pass

    def on_press(self, event):
        if not self.is_monitoring: return
        if event.keysym in self.pressed_keys: return
        self.pressed_keys.add(event.keysym)
        tok = self._map_keysym(event)
        if tok: self.current_events.append((time.perf_counter(), 'down', tok))

    def on_release(self, event):
        if not self.is_monitoring: return
        if event.keysym in self.pressed_keys: self.pressed_keys.remove(event.keysym)
        tok = self._map_keysym(event)
        if tok:
            self.current_events.append((time.perf_counter(), 'up', tok))
            # Sensitive check: 30 events (~15 chars)
            if len(self.current_events) >= 30:
                self._trigger_prediction()

    def _map_keysym(self, event):
        ch = event.char
        if ch and ch.isalnum(): return ch.lower()
        if event.keysym in ('Return', 'space'): return event.keysym.lower()
        return None

    def _trigger_prediction(self):
        events_to_process = list(self.current_events)
        self.current_events = []
        self.callback_status("Biometric Scrutiny...", ACCENT_BLUE)
        threading.Thread(target=self._predict_async, args=(events_to_process,), daemon=True).start()

    def _predict_async(self, events):
        if not self.voter: return
        feats = extract_features(events)
        if feats is None: 
            self.callback_status("Protection Active", SUCCESS_GREEN)
            return
        X_scaled = self.scaler.transform(feats.reshape(1, -1))
        prediction = self.voter.predict(X_scaled)[0]
        
        if prediction != self.target_username:
            self.callback_status("Security Alert!", ERROR_RED)
            self.callback_logout("Biometric Identity Mismatch!\nTyping characteristics do not match the authorized user.")
        else:
            self.callback_status("User Verified ✓", SUCCESS_GREEN)

class KeystrokeApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("TitanOffice ERM - Secure Suite")
        self.geometry("1100x750")
        self.configure(fg_color=BG_DARK)
        
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.current_user = None
        self.monitor = None

        for F in (LoginFrame, EnrollmentFrame, OfficeSuiteFrame):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("LoginFrame")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if page_name == "OfficeSuiteFrame":
            self.start_monitoring()
        else:
            self.stop_monitoring()

    def start_monitoring(self):
        if self.current_user:
            status_cb = self.frames["OfficeSuiteFrame"].update_status
            self.monitor = ContinuousAuthMonitor(self.force_logout, status_cb, self.current_user)
            self.monitor.is_monitoring = True
            self.bind_all('<KeyPress>', self.monitor.on_press)
            self.bind_all('<KeyRelease>', self.monitor.on_release)

    def stop_monitoring(self):
        if self.monitor:
            self.monitor.is_monitoring = False
            self.unbind_all('<KeyPress>')
            self.unbind_all('<KeyRelease>')
            self.monitor = None

    def force_logout(self, message):
        self.after(0, lambda: messagebox.showerror("Security Enforcement", message))
        self.after(100, lambda: self.show_frame("LoginFrame"))
        self.current_user = None

class LoginFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_DARK)
        self.controller = controller
        card = ctk.CTkFrame(self, fg_color=CARD_COLOR, width=420, height=500, corner_radius=25, border_width=1, border_color="#334155")
        card.place(relx=0.5, rely=0.5, anchor="center"); card.pack_propagate(False)
        ctk.CTkLabel(card, text="🛡️", font=ctk.CTkFont(size=70)).pack(pady=(45, 10))
        ctk.CTkLabel(card, text="TitanOffice", font=ctk.CTkFont(size=28, weight="bold")).pack()
        ctk.CTkLabel(card, text="CONTINUOUS BIOMETRIC ACCESS", font=ctk.CTkFont(size=11, weight="bold"), text_color=ACCENT_BLUE).pack(pady=(0, 25))
        self.entry_user = ctk.CTkEntry(card, width=320, height=45, placeholder_text="Username")
        self.entry_user.pack(pady=10)
        self.entry_pass = ctk.CTkEntry(card, width=320, height=45, placeholder_text="Password", show="*")
        self.entry_pass.pack(pady=10)
        ctk.CTkButton(card, text="Authorize", command=self.login, width=320, height=48, fg_color=ACCENT_BLUE, font=ctk.CTkFont(weight="bold")).pack(pady=25)
        ctk.CTkButton(card, text="Create Employee Account", font=ctk.CTkFont(size=12), fg_color="transparent", text_color=TEXT_SECONDARY, hover=False, command=lambda: controller.show_frame("EnrollmentFrame")).pack()

    def login(self):
        u, p = self.entry_user.get().strip(), self.entry_pass.get().strip()
        users = load_users()
        if u in users and users[u] == p:
            self.controller.current_user = u
            self.controller.show_frame("OfficeSuiteFrame")
        else: messagebox.showerror("Access Denied", "Invalid system credentials")

class EnrollmentFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_DARK)
        self.controller = controller
        self.enroll_samples = []
        self.enroll_target = 10
        self.enroll_text = "the quick brown fox jumps over the lazy dog 1234567890"
        self.current_events = []
        self.pressed_keys = set()
        
        card = ctk.CTkFrame(self, fg_color=CARD_COLOR, width=650, height=580, corner_radius=25)
        card.place(relx=0.5, rely=0.5, anchor="center"); card.pack_propagate(False)
        ctk.CTkLabel(card, text="Registration & Biometric Booting", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=30)
        
        info = ctk.CTkFrame(card, fg_color="transparent"); info.pack(pady=10)
        self.e_user = ctk.CTkEntry(info, width=280, height=42, placeholder_text="Assigned Username")
        self.e_user.grid(row=0, column=0, padx=10)
        self.e_pass = ctk.CTkEntry(info, width=280, height=42, placeholder_text="Secure Password", show="*")
        self.e_pass.grid(row=0, column=1, padx=10)

        self.lbl_einst = ctk.CTkLabel(card, text=f"Baseline (Type {self.enroll_target}x):\n\"{self.enroll_text}\"", text_color=TEXT_SECONDARY, font=ctk.CTkFont(slant="italic"), wraplength=550)
        self.lbl_einst.pack(pady=10)

        self.e_enroll = ctk.CTkEntry(card, width=580, height=48, placeholder_text="Type here to capture rhythm...")
        self.e_enroll.pack(pady=20)
        self.e_enroll.bind('<KeyPress>', self._on_key_press); self.e_enroll.bind('<KeyRelease>', self._on_key_release)

        self.pbar = ctk.CTkProgressBar(card, width=580, height=12, progress_color=ACCENT_BLUE)
        self.pbar.pack(); self.pbar.set(0)
        
        self.lbl_status = ctk.CTkLabel(card, text=f"Samples: 0 / {self.enroll_target}", font=ctk.CTkFont(weight="bold"), text_color=ACCENT_BLUE)
        self.lbl_status.pack(pady=10)
        ctk.CTkButton(card, text="Cancel", fg_color="transparent", text_color=ERROR_RED, command=lambda: controller.show_frame("LoginFrame")).pack(pady=10)

    def _on_key_press(self, event):
        if event.keysym in self.pressed_keys: return
        self.pressed_keys.add(event.keysym)
        ch = event.char
        if ch and ch.isalnum(): self.current_events.append((time.perf_counter(), 'down', ch.lower()))
        elif event.keysym == 'Return': self.current_events.append((time.perf_counter(), 'down', 'enter'))

    def _on_key_release(self, event):
        if event.keysym in self.pressed_keys: self.pressed_keys.remove(event.keysym)
        ch = event.char
        if ch and ch.isalnum(): self.current_events.append((time.perf_counter(), 'up', ch.lower()))
        elif event.keysym == 'Return':
            self.current_events.append((time.perf_counter(), 'up', 'enter'))
            self._process()

    def _process(self):
        evs = list(self.current_events); self.current_events = []; self.e_enroll.delete(0, 'end')
        feats = extract_features(evs)
        if feats is None: messagebox.showwarning("Incomplete", "Typing window too small."); return
        self.enroll_samples.append(feats)
        cnt = len(self.enroll_samples); self.pbar.set(cnt/self.enroll_target); self.lbl_status.configure(text=f"Samples: {cnt} / {self.enroll_target}")
        if cnt >= self.enroll_target: self._final()

    def _final(self):
        u, p = self.e_user.get().strip(), self.e_pass.get().strip()
        if not u or not p: messagebox.showerror("Fail", "Missing Account Info"); return
        users = load_users(); users[u] = p; save_users(users)
        self.lbl_status.configure(text="Conditioning AI Ensemble...")
        threading.Thread(target=self._train, args=(u, np.array(self.enroll_samples)), daemon=True).start()

    def _train(self, u, smp):
        try:
            s = synthesize_samples(smp, 600); csv = get_dataset_csv(); append_to_dataset(u, s, csv)
            X, Xt, y, yt, le, sc, _ = prepare_data(csv)
            v = EnsembleVoter(); v.train(X, y, le=le); v.save_all("models")
            joblib.dump(sc, "models/scaler.pkl"); joblib.dump(le, "models/label_encoder.pkl")
            self.controller.after(0, lambda: (messagebox.showinfo("Success", "Identity Recorded"), self.controller.show_frame("LoginFrame")))
        except Exception as e: self.controller.after(0, lambda: messagebox.showerror("System Error", str(e)))

class OfficeSuiteFrame(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=BG_DARK)
        self.controller = controller
        self.sidebar = ctk.CTkFrame(self, fg_color=SIDEBAR_COLOR, width=220, corner_radius=0)
        self.sidebar.pack(side="left", fill="y"); self.sidebar.pack_propagate(False)
        ctk.CTkLabel(self.sidebar, text="Titan ERP", font=ctk.CTkFont(size=22, weight="bold"), text_color=ACCENT_BLUE).pack(pady=30)
        menus = [("🏠 Dashboard", "Dash"), ("👥 HR Central", "HR"), ("📊 Projects", "Proj"), ("💸 Finance", "Fin"), ("⚙️ Settings", "Dash")]
        for label, sid in menus:
            ctk.CTkButton(self.sidebar, text=label, height=50, fg_color="transparent", anchor="w", font=ctk.CTkFont(size=14), command=lambda s=sid: self._switch_sub(s)).pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(self.sidebar, text="SECURED BY AI", font=ctk.CTkFont(size=10, weight="bold"), text_color="#334155").pack(side="bottom", pady=20)
        self.view_port = ctk.CTkFrame(self, fg_color="transparent"); self.view_port.pack(side="left", fill="both", expand=True)
        self.sub_frames = {}
        for SF in (DashView, HRView, ProjView, FinView):
            self.sub_frames[SF.__name__] = SF(self.view_port)
            self.sub_frames[SF.__name__].grid(row=0, column=0, sticky="nsew")
        self.view_port.grid_rowconfigure(0, weight=1); self.view_port.grid_columnconfigure(0, weight=1)
        self.footer = ctk.CTkFrame(self, fg_color=SIDEBAR_COLOR, height=35, corner_radius=0); self.footer.pack(side="bottom", fill="x")
        self.lbl_s_dot = ctk.CTkLabel(self.footer, text="●", text_color=SUCCESS_GREEN); self.lbl_s_dot.pack(side="left", padx=(25, 5))
        self.lbl_s_txt = ctk.CTkLabel(self.footer, text="System Integrity: High", font=ctk.CTkFont(size=11)); self.lbl_s_txt.pack(side="left")
        self._switch_sub("Dash")

    def _switch_sub(self, sid):
        frame_map = {"Dash": "DashView", "HR": "HRView", "Proj": "ProjView", "Fin": "FinView"}
        self.sub_frames[frame_map[sid]].tkraise()

    def update_status(self, text, color):
        self.after(0, lambda: (self.lbl_s_txt.configure(text=f"AI Monitor: {text}"), self.lbl_s_dot.configure(text_color=color)))

class DashView(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        ctk.CTkLabel(self, text="Enterprise Dashboard", font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=30, pady=30)
        stat_f = ctk.CTkFrame(self, fg_color="transparent"); stat_f.pack(fill="x", padx=30)
        for i, (t, v) in enumerate([("Total Staff", "1,248"), ("Active Projects", "154"), ("Monthly Revenue", "$2.8M")]):
            c = ctk.CTkFrame(stat_f, fg_color=CARD_COLOR, width=220, height=120, corner_radius=15)
            c.grid(row=0, column=i, padx=10); c.pack_propagate(False)
            ctk.CTkLabel(c, text=t, text_color=TEXT_SECONDARY).pack(pady=(20, 5))
            ctk.CTkLabel(c, text=v, font=ctk.CTkFont(size=24, weight="bold")).pack()

class HRView(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        ctk.CTkLabel(self, text="Human Resources Central", font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=40, pady=30)
        grid = ctk.CTkFrame(self, fg_color="transparent"); grid.pack(fill="both", expand=True, padx=40)
        fields = ["Legal First Name", "Legal Last Name", "Employee ID UID", "Department Code", "Office Extension", "Corporate Email"]
        for i, f in enumerate(fields):
            fr = ctk.CTkFrame(grid, fg_color="transparent"); fr.grid(row=i//2, column=i%2, padx=15, pady=10, sticky="w")
            ctk.CTkLabel(fr, text=f, text_color=TEXT_SECONDARY, font=ctk.CTkFont(size=12)).pack(anchor="w")
            ctk.CTkEntry(fr, width=320, height=38).pack(pady=5)

class ProjView(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        ctk.CTkLabel(self, text="Project Management Workspace", font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=40, pady=30)
        ctk.CTkLabel(self, text="Project Scope & Technical Description:", text_color=TEXT_SECONDARY).pack(anchor="w", padx=45)
        ctk.CTkTextbox(self, height=300).pack(fill="x", padx=45, pady=10)

class FinView(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent, fg_color="transparent")
        ctk.CTkLabel(self, text="Expense & Finance Submission", font=ctk.CTkFont(size=22, weight="bold")).pack(anchor="w", padx=40, pady=30)
        grid = ctk.CTkFrame(self, fg_color="transparent"); grid.pack(padx=40, fill="x")
        for i, f in enumerate(["Vendor Name", "Invoice Number", "Amount ($)", "Category (Travel/Ops)", "Approval Manager"]):
            fr = ctk.CTkFrame(grid, fg_color="transparent"); fr.pack(fill="x", pady=8)
            ctk.CTkLabel(fr, text=f, text_color=TEXT_SECONDARY, width=150, anchor="w").pack(side="left")
            ctk.CTkEntry(fr, width=420, height=38).pack(side="left")

if __name__ == "__main__":
    app = KeystrokeApp()
    app.mainloop()
