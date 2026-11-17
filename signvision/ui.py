import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

from PIL import ImageTk


class SignVisionUI:
    def __init__(self, root: tk.Tk, on_start: Callable, on_stop: Callable, on_clear: Callable, on_close: Callable,
                 on_mode: Callable | None = None, on_word_action: Callable | None = None):
        self.root, self.on_start, self.on_stop, self.on_clear, self.on_close = root, on_start, on_stop, on_clear, on_close
        self.on_mode = on_mode or (lambda _mode: None)
        self.on_word_action = on_word_action or (lambda _action: None)
        root.title("SignVision AI - Real-Time Hand Gesture Recognition")
        root.geometry("1000x760")
        root.configure(bg="#0b1220")
        root.protocol("WM_DELETE_WINDOW", on_close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#111827")
        style.configure("TLabel", background="#111827", foreground="#f8fafc")
        style.configure("Header.TLabel", font=("Segoe UI", 18, "bold"), foreground="#67e8f9")
        style.configure("Result.TLabel", font=("Segoe UI", 18, "bold"), foreground="#a7f3d0")
        self.video = ttk.Label(root, text="Camera is stopped", anchor=tk.CENTER)
        self.video.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        ttk.Label(root, text="SIGNVISION AI  |  Real-Time Hand Gesture Recognition", style="Header.TLabel").pack()
        self.result = ttk.Label(root, text="Gesture: —\nInterpretation: —", style="Result.TLabel")
        self.result.pack(pady=4)
        self.status = ttk.Label(root, text="Ready")
        self.status.pack()
        buttons = ttk.Frame(root); buttons.pack(pady=8)
        ttk.Button(buttons, text="Start", command=on_start).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Stop", command=on_stop).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Clear history", command=on_clear).pack(side=tk.LEFT, padx=4)
        mode = ttk.Frame(root); mode.pack(pady=(0, 6))
        ttk.Label(mode, text="Analysis mode:").pack(side=tk.LEFT)
        self.mode = tk.StringVar(value="alphabet")
        ttk.Radiobutton(mode, text="Alphabet", variable=self.mode, value="alphabet",
                        command=lambda: self.on_mode(self.mode.get())).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(mode, text="Word", variable=self.mode, value="word",
                        command=lambda: self.on_mode(self.mode.get())).pack(side=tk.LEFT, padx=4)
        word = ttk.Frame(root); word.pack(pady=(0, 6))
        for label, action in (
            ("START WORD ANALYSIS", "start"), ("Pause", "pause"), ("Resume", "resume"),
            ("Add letter", "add"), ("Space", "space"), ("Undo letter", "backspace"),
            ("Clear word", "clear"), ("Finish word", "finish"),
        ):
            ttk.Button(word, text=label, command=lambda a=action: self.on_word_action(a)).pack(
                side=tk.LEFT, padx=2
            )
        self.word = ttk.Label(root, text="Word: —", style="Result.TLabel")
        self.word.pack(pady=(0, 4))
        self.history = tk.Listbox(root, height=6)
        self.history.pack(fill=tk.X, padx=12, pady=(0, 12))

    def set_image(self, image) -> None:
        photo = ImageTk.PhotoImage(image)
        self.video.configure(image=photo, text="")
        self.video.image = photo

    def set_result(self, gesture: str, meaning: str, score: float) -> None:
        confidence = f"{score:.0%}" if gesture not in {"NO HAND DETECTED", "UNCERTAIN"} else "—"
        self.result.configure(
            text=f"Detected gesture: {gesture}\nInterpretation: {meaning}   Confidence: {confidence}"
        )

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def set_word(self, text: str) -> None:
        self.word.configure(text=f"Word: {text or '—'}")

    def show_error(self, text: str) -> None:
        self.set_status(text)
        messagebox.showerror("SignVision error", text)

    def refresh_history(self, entries) -> None:
        self.history.delete(0, tk.END)
        for item in reversed(entries):
            self.history.insert(tk.END, f"{item.timestamp:%H:%M:%S}  {item.label} ({item.score:.0%})")
