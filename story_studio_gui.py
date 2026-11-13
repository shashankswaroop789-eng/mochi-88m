"""
Story Studio GUI - ModernGPT 88.1M Desktop Application
Provides a complete, modern graphical interface for autonomous story generation.
No terminal required: launch via double-click on Windows.
"""

import os
import sys
import time
import re
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# Enable High-DPI sharpness on Windows
try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Model paths - Primary: newly trained 30,000-story master checkpoint
MODEL_PATH = os.path.join(BASE_DIR, "hf_mochi88m_master.pt")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "hf_master_gpt88m.pt")
TOKENIZER_PATH = os.path.join(BASE_DIR, "spm_hf_master.model")
OUTPUT_DIR = os.path.join(BASE_DIR, "novel_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Coherent Short Story Presets (High Coherence on 88.1M Architecture)
PRESETS = {
    "🚗 Shashank & Rajesh": (
        "Write a short story about Shashank, a proud young mechanic working in a highway tin shed, "
        "and Rajesh Khanna, a wealthy businessman whose luxury car breaks down on the desert road."
    ),
    "👑 Savita & Karthik": (
        "Write a short story about Savita Rajput, the proud family matriarch, confronting her grandson Karthik "
        "inside the ancestral mansion about his loyalty to the family name."
    ),
    "🤝 Avinash & Shashank": (
        "Write a short story about Avinash Rajput quietly visiting his estranged son Shashank at his dusty "
        "roadside workshop after years of bitter silence."
    ),
    "💎 Samira's Pride": (
        "Write a short story about Samira Khanna refusing to enter an arranged dynastic marriage for corporate power, "
        "vowing to make her own destiny."
    ),
    "📜 Full Rajput Dynasty": (
        "Write a story of the Rajput family. Savita Rajput gathers her family inside Rajput Sadan to announce "
        "the golden alliance with the Khanna Empire, while thirty miles away on the highway bypass, her outcast "
        "grandson Shashank works in a rusted tin shed as an honest mechanic."
    )
}

DEFAULT_PROMPT = PRESETS["🚗 Shashank & Rajesh"]

# --- THEME COLORS ---
BG_DARK = "#12131C"
BG_CARD = "#1B1D2A"
BG_INPUT = "#161822"
BG_SIDEBAR = "#181926"
ACCENT_PRIMARY = "#6366F1"       # Indigo
ACCENT_HOVER = "#4F46E5"
ACCENT_SUCCESS = "#10B981"       # Emerald green
ACCENT_SUCCESS_HOVER = "#059669"
ACCENT_DANGER = "#EF4444"        # Red
ACCENT_DANGER_HOVER = "#DC2626"
BORDER_COLOR = "#2C2F44"
TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#94A3B8"
TEXT_MUTED = "#64748B"


class StoryStudioGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🍡 Mochi Story Studio - Mochi-88M Story Generator")
        self.geometry("1240x860")
        self.minsize(1020, 720)
        self.configure(bg=BG_DARK)

        # State
        self.model = None
        self.sp = None
        self.device = "cpu"
        self.is_generating = False
        self.stop_requested = False
        self.current_chapter = 1
        self.chapters_data = {}
        self.cumulative_story = ""
        self.queue = queue.Queue()

        self._build_ui()
        self._start_queue_poller()
        self._async_load_model()

    def _build_ui(self):
        # 1. Top Header
        header = tk.Frame(self, bg=BG_CARD, height=64, padx=20, pady=12, highlightthickness=1, highlightbackground=BORDER_COLOR)
        header.pack(fill=tk.X, side=tk.TOP)

        title_frame = tk.Frame(header, bg=BG_CARD)
        title_frame.pack(side=tk.LEFT)

        app_title = tk.Label(
            title_frame, 
            text="🍡 Mochi Story Studio", 
            font=("Segoe UI", 16, "bold"), 
            fg=TEXT_PRIMARY, 
            bg=BG_CARD
        )
        app_title.pack(side=tk.LEFT)

        app_subtitle = tk.Label(
            title_frame, 
            text=" | Mochi-88M Coherent Story Engine", 
            font=("Segoe UI", 11), 
            fg=TEXT_SECONDARY, 
            bg=BG_CARD
        )
        app_subtitle.pack(side=tk.LEFT, padx=(6, 0))

        # Status Badge on Header Right
        self.status_badge = tk.Label(
            header,
            text="● Loading Engine...",
            font=("Segoe UI", 10, "bold"),
            fg="#F59E0B",
            bg="#2B271A",
            padx=12,
            pady=4
        )
        self.status_badge.pack(side=tk.RIGHT)

        # 2. Main Content Split View (Left = Controls & Setup, Right = Live Viewer & Chat Loop)
        main_pane = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG_DARK, sashwidth=4, sashrelief="flat")
        main_pane.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

        # --- LEFT PANEL: Settings & Story Prompt ---
        left_frame = tk.Frame(main_pane, bg=BG_SIDEBAR, padx=16, pady=16, highlightthickness=1, highlightbackground=BORDER_COLOR)
        main_pane.add(left_frame, minsize=420, width=470)

        # Section: Story Premise / Outline
        prompt_header = tk.Frame(left_frame, bg=BG_SIDEBAR)
        prompt_header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            prompt_header, 
            text="Story Premise & Characters", 
            font=("Segoe UI", 12, "bold"), 
            fg=TEXT_PRIMARY, 
            bg=BG_SIDEBAR
        ).pack(side=tk.LEFT)

        # Preset Buttons Row
        preset_bar = tk.Frame(left_frame, bg=BG_SIDEBAR)
        preset_bar.pack(fill=tk.X, pady=(0, 8))

        tk.Label(
            preset_bar, 
            text="Presets:", 
            font=("Segoe UI", 9, "bold"), 
            fg=TEXT_MUTED, 
            bg=BG_SIDEBAR
        ).pack(side=tk.LEFT, padx=(0, 4))

        for name in ["🚗 Shashank & Rajesh", "👑 Savita & Karthik", "🤝 Avinash & Shashank", "💎 Samira's Pride"]:
            btn = tk.Button(
                preset_bar,
                text=name,
                font=("Segoe UI", 8),
                fg=TEXT_PRIMARY,
                bg="#25273C",
                activebackground="#373B5B",
                activeforeground="#A5B4FC",
                relief="flat",
                padx=6,
                pady=2,
                cursor="hand2",
                command=lambda p=name: self._load_preset(p)
            )
            btn.pack(side=tk.LEFT, padx=2)

        # Prompt Text Area
        self.prompt_text = tk.Text(
            left_frame,
            font=("Segoe UI", 10),
            fg=TEXT_PRIMARY,
            bg=BG_INPUT,
            insertbackground="white",
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_PRIMARY,
            wrap=tk.WORD,
            height=10
        )
        self.prompt_text.pack(fill=tk.BOTH, expand=True, pady=(0, 12))
        self.prompt_text.insert("1.0", DEFAULT_PROMPT)

        # Section: Generation Settings Card
        settings_card = tk.LabelFrame(
            left_frame, 
            text=" Generation Settings ", 
            font=("Segoe UI", 10, "bold"), 
            fg=TEXT_SECONDARY, 
            bg=BG_CARD,
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground=BORDER_COLOR
        )
        settings_card.pack(fill=tk.X, pady=(0, 14))

        # Words Slider (Optimized for 88.1M short stories)
        lbl_words_row = tk.Frame(settings_card, bg=BG_CARD)
        lbl_words_row.pack(fill=tk.X, pady=(2, 2))
        tk.Label(lbl_words_row, text="Target Words per Scene:", font=("Segoe UI", 9), fg=TEXT_SECONDARY, bg=BG_CARD).pack(side=tk.LEFT)
        self.lbl_words_val = tk.Label(lbl_words_row, text="250 words", font=("Segoe UI", 9, "bold"), fg=ACCENT_PRIMARY, bg=BG_CARD)
        self.lbl_words_val.pack(side=tk.RIGHT)

        self.slider_words = ttk.Scale(
            settings_card, 
            from_=100, 
            to=450, 
            value=250, 
            command=lambda v: self.lbl_words_val.config(text=f"{int(float(v)):,} words")
        )
        self.slider_words.pack(fill=tk.X, pady=(0, 8))

        # Creativity / Temperature
        lbl_temp_row = tk.Frame(settings_card, bg=BG_CARD)
        lbl_temp_row.pack(fill=tk.X, pady=(2, 2))
        tk.Label(lbl_temp_row, text="Creativity (Temperature):", font=("Segoe UI", 9), fg=TEXT_SECONDARY, bg=BG_CARD).pack(side=tk.LEFT)
        self.lbl_temp_val = tk.Label(lbl_temp_row, text="0.70", font=("Segoe UI", 9, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
        self.lbl_temp_val.pack(side=tk.RIGHT)

        self.slider_temp = ttk.Scale(
            settings_card, 
            from_=0.3, 
            to=1.05, 
            value=0.70, 
            command=lambda v: self.lbl_temp_val.config(text=f"{float(v):.2f}")
        )
        self.slider_temp.pack(fill=tk.X, pady=(0, 8))

        # Repetition Penalty (Optimal range: 1.05 - 1.20)
        lbl_rep_row = tk.Frame(settings_card, bg=BG_CARD)
        lbl_rep_row.pack(fill=tk.X, pady=(2, 2))
        tk.Label(lbl_rep_row, text="Repetition Penalty:", font=("Segoe UI", 9), fg=TEXT_SECONDARY, bg=BG_CARD).pack(side=tk.LEFT)
        self.lbl_rep_val = tk.Label(lbl_rep_row, text="1.12", font=("Segoe UI", 9, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
        self.lbl_rep_val.pack(side=tk.RIGHT)

        self.slider_rep = ttk.Scale(
            settings_card, 
            from_=1.0, 
            to=1.28, 
            value=1.12, 
            command=lambda v: self.lbl_rep_val.config(text=f"{float(v):.2f}")
        )
        self.slider_rep.pack(fill=tk.X, pady=(0, 4))

        # Primary Buttons (Generate Story / Stop)
        btn_frame = tk.Frame(left_frame, bg=BG_SIDEBAR)
        btn_frame.pack(fill=tk.X, pady=(4, 0))

        self.btn_generate = tk.Button(
            btn_frame,
            text="✨ Generate Story",
            font=("Segoe UI", 11, "bold"),
            fg="white",
            bg=ACCENT_PRIMARY,
            activebackground=ACCENT_HOVER,
            activeforeground="white",
            relief="flat",
            pady=10,
            cursor="hand2",
            state=tk.DISABLED,
            command=self._on_click_start_story
        )
        self.btn_generate.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.btn_stop = tk.Button(
            btn_frame,
            text="⏹ Stop",
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg=ACCENT_DANGER,
            activebackground=ACCENT_DANGER_HOVER,
            activeforeground="white",
            relief="flat",
            pady=10,
            padx=14,
            cursor="hand2",
            state=tk.DISABLED,
            command=self._on_click_stop
        )
        self.btn_stop.pack(side=tk.RIGHT)

        # --- RIGHT PANEL: Interactive Reader & Chat Progression ---
        right_frame = tk.Frame(main_pane, bg=BG_DARK, padx=8, pady=0)
        main_pane.add(right_frame, minsize=540)

        # Right Top: Chapter Selector & Actions Bar
        top_bar = tk.Frame(right_frame, bg=BG_CARD, padx=14, pady=10, highlightthickness=1, highlightbackground=BORDER_COLOR)
        top_bar.pack(fill=tk.X, pady=(0, 8))

        tk.Label(top_bar, text="View Story / Scene:", font=("Segoe UI", 10, "bold"), fg=TEXT_SECONDARY, bg=BG_CARD).pack(side=tk.LEFT, padx=(0, 8))

        self.chapter_var = tk.StringVar(value="Scene 1 (Pending)")
        self.chapter_dropdown = ttk.Combobox(
            top_bar, 
            textvariable=self.chapter_var, 
            state="readonly", 
            width=24,
            font=("Segoe UI", 9)
        )
        self.chapter_dropdown.pack(side=tk.LEFT, padx=(0, 12))
        self.chapter_dropdown.bind("<<ComboboxSelected>>", self._on_chapter_selected)

        # Right Actions
        btn_copy = tk.Button(
            top_bar,
            text="📋 Copy Text",
            font=("Segoe UI", 9),
            fg=TEXT_PRIMARY,
            bg="#2E3349",
            activebackground="#3B425E",
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._copy_current_chapter
        )
        btn_copy.pack(side=tk.RIGHT, padx=(4, 0))

        btn_save_current = tk.Button(
            top_bar,
            text="💾 Save Story",
            font=("Segoe UI", 9),
            fg=TEXT_PRIMARY,
            bg="#2E3349",
            activebackground="#3B425E",
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
            command=self._save_current_chapter
        )
        btn_save_current.pack(side=tk.RIGHT, padx=(4, 0))

        # Main Story Display Text Box
        text_container = tk.Frame(right_frame, bg=BG_INPUT, highlightthickness=1, highlightbackground=BORDER_COLOR)
        text_container.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        scrollbar = ttk.Scrollbar(text_container, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.story_display = tk.Text(
            text_container,
            font=("Georgia", 11),
            fg="#F1F5F9",
            bg=BG_INPUT,
            insertbackground="white",
            relief="flat",
            padx=20,
            pady=16,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set
        )
        self.story_display.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.story_display.yview)

        # Interactive "Next Chapter Instruction" Card
        self.next_ch_box = tk.LabelFrame(
            right_frame,
            text=" 📖 Scene Complete! What should happen next? ",
            font=("Segoe UI", 10, "bold"),
            fg=ACCENT_SUCCESS,
            bg=BG_CARD,
            padx=14,
            pady=10,
            highlightthickness=1,
            highlightbackground="#065F46"
        )

        # Next chapter instruction entry
        self.txt_next_instruction = tk.Entry(
            self.next_ch_box,
            font=("Segoe UI", 10),
            fg=TEXT_PRIMARY,
            bg=BG_INPUT,
            insertbackground="white",
            relief="flat",
            highlightthickness=1,
            highlightbackground=BORDER_COLOR,
            highlightcolor=ACCENT_SUCCESS
        )
        self.txt_next_instruction.pack(fill=tk.X, pady=(2, 8))
        self.txt_next_instruction.bind("<Return>", lambda e: self._on_click_next_chapter_custom())

        # Buttons in Next Chapter Box
        next_btns_row = tk.Frame(self.next_ch_box, bg=BG_CARD)
        next_btns_row.pack(fill=tk.X)

        self.btn_next_custom = tk.Button(
            next_btns_row,
            text="✨ Generate Next Scene with Instructions",
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg=ACCENT_SUCCESS,
            activebackground=ACCENT_SUCCESS_HOVER,
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._on_click_next_chapter_custom
        )
        self.btn_next_custom.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_next_auto = tk.Button(
            next_btns_row,
            text="🎲 Continue Naturally",
            font=("Segoe UI", 9),
            fg=TEXT_PRIMARY,
            bg="#2E3349",
            activebackground="#3B425E",
            relief="flat",
            padx=14,
            pady=6,
            cursor="hand2",
            command=self._on_click_next_chapter_auto
        )
        self.btn_next_auto.pack(side=tk.LEFT)

        # 3. Bottom Status Bar
        status_bar = tk.Frame(self, bg="#0C0D14", height=28, padx=16, pady=4)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_status = tk.Label(
            status_bar, 
            text="Initializing Story Studio...", 
            font=("Segoe UI", 9), 
            fg=TEXT_MUTED, 
            bg="#0C0D14"
        )
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_progress = tk.Label(
            status_bar, 
            text="Words: 0 / 0 | Speed: 0 tok/s", 
            font=("Segoe UI", 9, "bold"), 
            fg=TEXT_SECONDARY, 
            bg="#0C0D14"
        )
        self.lbl_progress.pack(side=tk.RIGHT)

    # --- MODEL LOADING (BACKGROUND THREAD) ---
    def _async_load_model(self):
        def worker():
            try:
                import torch
                import sentencepiece as spm
                from modern_transformer import ModernGPT

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.device = device
                self.queue.put(("status", f"Loading weights into {device.upper()}..."))

                load_path = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL_PATH
                if not os.path.exists(load_path) or not os.path.exists(TOKENIZER_PATH):
                    self.queue.put(("status", "Downloading Mochi-88M weights from Hugging Face..."))
                    import subprocess
                    subprocess.run([sys.executable, os.path.join(BASE_DIR, "download_model.py")], check=True)
                    load_path = MODEL_PATH

                sp = spm.SentencePieceProcessor()
                sp.load(TOKENIZER_PATH)

                ckpt = torch.load(load_path, map_location=device, weights_only=False)
                config = ckpt.get("config", {})
                
                model = ModernGPT(
                    vocab_size=config.get("vocab_size", 4096),
                    dim=config.get("dim", 768),
                    n_layers=config.get("n_layers", 12),
                    n_heads=config.get("n_heads", 12),
                    max_seq_len=config.get("max_seq_len", 512),
                    dropout=0.0
                ).to(device)

                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                model.load_state_dict(state_dict)
                model.eval()

                self.model = model
                self.sp = sp

                gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
                model_name = os.path.basename(load_path)
                self.queue.put(("loaded", (f"Ready | 🍡 Mochi-88M ({gpu_name})", gpu_name, model_name)))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.queue.put(("load_error", f"Failed to load model: {str(e)}"))

        threading.Thread(target=worker, daemon=True).start()

    # --- QUEUE POLLER FOR THREAD-SAFE UI UPDATES ---
    def _start_queue_poller(self):
        def check():
            try:
                while True:
                    msg_type, data = self.queue.get_nowait()
                    if msg_type == "status":
                        self.lbl_status.config(text=data)
                    elif msg_type == "loaded":
                        status_text, gpu_name, model_name = data
                        self.lbl_status.config(text=f"{status_text} | Checkpoint: {model_name}")
                        self.status_badge.config(text=f"● Ready ({gpu_name})", fg="#34D399", bg="#064E3B")
                        self.btn_generate.config(state=tk.NORMAL)
                    elif msg_type == "load_error":
                        self.lbl_status.config(text=data)
                        self.status_badge.config(text="● Load Error", fg="#EF4444", bg="#451A1A")
                        self.btn_generate.config(state=tk.DISABLED)
                        messagebox.showerror("Model Loading Error", data)
                    elif msg_type == "stream":
                        chunk, current_words, target_words, speed = data
                        self.story_display.insert(tk.END, chunk)
                        self.story_display.see(tk.END)
                        self.lbl_progress.config(
                            text=f"Words: {current_words:,} / {target_words:,} | Speed: {speed:.1f} tok/s"
                        )
                    elif msg_type == "chapter_done":
                        ch_num, full_text, words = data
                        self._finish_chapter_ui(ch_num, full_text, words)
                    elif msg_type == "error":
                        messagebox.showerror("Generation Error", data)
                        self.lbl_status.config(text="Generation halted due to error.")
                        self._reset_generation_ui()
            except queue.Empty:
                pass
            self.after(40, check)
        self.after(40, check)

    # --- USER ACTIONS ---
    def _load_preset(self, preset_name):
        if preset_name in PRESETS:
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", PRESETS[preset_name])
            self.slider_words.set(300)
            self.lbl_words_val.config(text="300 words")

    def _on_click_start_story(self):
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Empty Prompt", "Please enter a story premise or characters first.")
            return

        self.current_chapter = 1
        self.chapters_data.clear()
        self.cumulative_story = ""
        self.story_display.delete("1.0", tk.END)
        formatted_prompt = f"### Prompt:\n{prompt}\n\n### Story:\n"
        self._start_generation_worker(formatted_prompt, chapter_num=1)

    def _on_click_next_chapter_custom(self):
        instruction = self.txt_next_instruction.get().strip()
        if not instruction:
            messagebox.showinfo("Instructions Needed", "Type what should happen, or click 'Continue Naturally'.")
            return
        
        next_ch = self.current_chapter + 1
        prev_tail = self.chapters_data.get(self.current_chapter, "")[-350:]
        
        formatted_prompt = (
            f"### Prompt:\n"
            f"Previous scene ending:\n{prev_tail}\n\n"
            f"Next scene:\n{instruction}\n\n"
            f"### Story:\n"
        )
        self._start_generation_worker(formatted_prompt, chapter_num=next_ch)

    def _on_click_next_chapter_auto(self):
        next_ch = self.current_chapter + 1
        prev_tail = self.chapters_data.get(self.current_chapter, "")[-350:]
        formatted_prompt = (
            f"### Prompt:\n"
            f"Continue the story naturally from this scene:\n{prev_tail}\n\n"
            f"### Story:\n"
        )
        self._start_generation_worker(formatted_prompt, chapter_num=next_ch)

    def _on_click_stop(self):
        if self.is_generating:
            self.stop_requested = True
            self.lbl_status.config(text="Stopping generation...")
            self.btn_stop.config(state=tk.DISABLED)

    # --- GENERATION WORKER ---
    def _start_generation_worker(self, prompt_text, chapter_num):
        if self.model is None or self.sp is None:
            messagebox.showwarning(
                "Model Still Loading", 
                "The 88.1M model is still loading into GPU memory. Please wait until the badge shows '● Ready'."
            )
            return

        self.is_generating = True
        self.stop_requested = False
        self.current_chapter = chapter_num

        target_words = int(self.slider_words.get())
        temperature = float(self.slider_temp.get())
        repetition_penalty = float(self.slider_rep.get())

        # UI state change
        self.btn_generate.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.next_ch_box.pack_forget()
        self.lbl_status.config(text=f"Generating Scene {chapter_num} with 🍡 Mochi-88M...")
        
        # Display scene header in text box
        self.story_display.delete("1.0", tk.END)
        header_text = f"========================================\n  SCENE {chapter_num}\n========================================\n\n"
        self.story_display.insert(tk.END, header_text)

        def worker():
            try:
                import torch

                token_ids = self.sp.encode_as_ids(prompt_text)
                if not token_ids:
                    token_ids = [0]

                # Intelligent prompt compression if prompt is too long for 512 context
                # Keep prompt within 280 tokens so the model has ample context for creative generation
                if len(token_ids) > 280:
                    token_ids = token_ids[:40] + token_ids[-240:]

                all_tokens = list(token_ids)
                initial_len = len(all_tokens)
                last_text_len = 0
                
                t0 = time.time()
                total_tokens_generated = 0

                while True:
                    if self.stop_requested:
                        break

                    # Context window: keep last 384 tokens
                    context_ids = all_tokens[-384:]
                    idx = torch.tensor([context_ids], dtype=torch.long, device=self.device)
                    # Calculate how many prompt tokens remain in current context window
                    prompt_in_ctx = max(0, initial_len - (len(all_tokens) - len(context_ids)))

                    with torch.no_grad():
                        output_tokens = self.model.fast_generate(
                            idx,
                            max_new_tokens=20,
                            temperature=temperature,
                            top_k=40,
                            top_p=0.88,
                            repetition_penalty=repetition_penalty,
                            prompt_len=prompt_in_ctx,
                            no_repeat_ngram_size=4,
                            stop_on_eos=True
                        )

                    new_ids = output_tokens[0].tolist()[len(context_ids):]
                    if not new_ids:
                        break

                    all_tokens.extend(new_ids)
                    total_tokens_generated += len(new_ids)

                    # Stop naturally if EOS was generated
                    if len(new_ids) < 20 or (new_ids and new_ids[-1] == 2):
                        break

                    # Decode only generated story text (after initial prompt)
                    gen_story_tokens = all_tokens[initial_len:]
                    current_text = self.sp.decode(gen_story_tokens)
                    current_words = len(current_text.split())

                    # Stream delta text to UI
                    new_chunk = current_text[last_text_len:]
                    last_text_len = len(current_text)

                    elapsed = max(0.1, time.time() - t0)
                    speed = total_tokens_generated / elapsed

                    self.queue.put(("stream", (new_chunk, current_words, target_words, speed)))

                    if current_words >= target_words:
                        break

                # Finish up and trim cleanly to sentence boundary
                full_gen_text = self.sp.decode(all_tokens[initial_len:])
                full_gen_text = re.sub(r'\n{3,}', '\n\n', full_gen_text)
                
                # Sentence end trim
                match = re.search(r'[\.\!\?]["\']?\s*$', full_gen_text)
                if not match:
                    last_punct = max(full_gen_text.rfind('.'), full_gen_text.rfind('!'), full_gen_text.rfind('?'))
                    if last_punct != -1:
                        full_gen_text = full_gen_text[:last_punct + 1].strip()

                final_words = len(full_gen_text.split())

                # Save to disk
                filepath = os.path.join(OUTPUT_DIR, f"scene_{chapter_num:02d}.txt")
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"SCENE {chapter_num}\n\n" + full_gen_text)

                self.queue.put(("chapter_done", (chapter_num, full_gen_text, final_words)))

            except Exception as e:
                import traceback
                traceback.print_exc()
                self.queue.put(("error", f"Generation Error: {str(e)}"))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_chapter_ui(self, ch_num, full_text, words):
        self.is_generating = False
        self.stop_requested = False
        self.chapters_data[ch_num] = full_text

        # Update dropdown list
        vals = [f"Scene {n} ({len(self.chapters_data[n].split()):,} words)" for n in sorted(self.chapters_data.keys())]
        self.chapter_dropdown.config(values=vals)
        self.chapter_dropdown.current(len(vals) - 1)

        # Update status
        self.lbl_status.config(text=f"Scene {ch_num} Complete! ({words:,} words) - Saved to novel_output/")
        self.btn_generate.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

        # Show Next Chapter Dialogue Box
        next_num = ch_num + 1
        self.next_ch_box.config(
            text=f" 📖 Scene {ch_num} Complete! What should happen in Scene {next_num}? "
        )
        self.txt_next_instruction.delete(0, tk.END)
        self.btn_next_custom.config(text=f"✨ Generate Scene {next_num} with Instructions")
        self.btn_next_auto.config(text=f"🎲 Continue Scene {next_num} Naturally")
        self.next_ch_box.pack(fill=tk.X, pady=(6, 0))
        self.txt_next_instruction.focus_set()

    def _reset_generation_ui(self):
        self.is_generating = False
        self.stop_requested = False
        if self.model is not None:
            self.btn_generate.config(state=tk.NORMAL)
        else:
            self.btn_generate.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.DISABLED)

    # --- CHAPTER SELECTION & EXPORT ---
    def _on_chapter_selected(self, event=None):
        idx = self.chapter_dropdown.current()
        if idx >= 0:
            ch_num = sorted(self.chapters_data.keys())[idx]
            text = self.chapters_data[ch_num]
            self.story_display.delete("1.0", tk.END)
            self.story_display.insert(
                tk.END, 
                f"========================================\n  SCENE {ch_num}\n========================================\n\n"
            )
            self.story_display.insert(tk.END, text)

    def _copy_current_chapter(self):
        text = self.story_display.get("1.0", tk.END).strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            messagebox.showinfo("Copied", "Story text copied to clipboard!")

    def _save_current_chapter(self):
        text = self.story_display.get("1.0", tk.END).strip()
        if not text:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Markdown files", "*.md")],
            initialdir=OUTPUT_DIR,
            initialfile=f"scene_{self.current_chapter:02d}.txt"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            messagebox.showinfo("Saved", f"Story saved successfully to:\n{path}")


def main():
    app = StoryStudioGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
