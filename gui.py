from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from analysis_pipeline import analyze_pdf
from config import OUTPUT_CSV, OUTPUT_JSON, OUTPUT_MARKED_PDF
from pdf_colored_report import create_colored_pdf_report


APP_TITLE = "LeakShield - Sensitive Information Leakage Detection"

# ==========================================================
# Visual system
# ==========================================================

COLOR_BG = "#EEF3F9"
COLOR_SURFACE = "#FFFFFF"
COLOR_SURFACE_ALT = "#F8FAFC"
COLOR_HEADER = "#0B1F3A"
COLOR_PRIMARY = "#1D4ED8"
COLOR_PRIMARY_HOVER = "#1E40AF"
COLOR_PRIMARY_SOFT = "#DBEAFE"
COLOR_ACCENT = "#06B6D4"
COLOR_TEXT = "#0F172A"
COLOR_MUTED = "#64748B"
COLOR_BORDER = "#D7E0EA"
COLOR_SUCCESS = "#15803D"
COLOR_SUCCESS_BG = "#DCFCE7"
COLOR_WARNING = "#B45309"
COLOR_WARNING_BG = "#FEF3C7"
COLOR_ERROR = "#B91C1C"
COLOR_ERROR_BG = "#FEE2E2"
COLOR_DISABLED = "#CBD5E1"

FONT_FAMILY = "Segoe UI"


class LeakShieldGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1240x820")
        self.root.minsize(1020, 700)
        self.root.resizable(True, True)
        self.root.configure(bg=COLOR_BG)

        self.selected_pdf_path = tk.StringVar(value="No PDF selected")
        self.status_text = tk.StringVar(value="Ready")
        self.progress_text = tk.StringVar(value="Waiting for a document")
        self.progress_value = tk.DoubleVar(value=0)

        self.is_analyzing = False

        self._configure_styles()
        self._build_ui()

    # ======================================================
    # Styling
    # ======================================================

    def _configure_styles(self):
        style = ttk.Style(self.root)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "LeakShield.Horizontal.TProgressbar",
            troughcolor="#E2E8F0",
            background=COLOR_PRIMARY,
            lightcolor=COLOR_PRIMARY,
            darkcolor=COLOR_PRIMARY,
            bordercolor="#E2E8F0",
            thickness=12,
        )

        style.configure(
            "LeakShield.Vertical.TScrollbar",
            background="#CBD5E1",
            troughcolor=COLOR_SURFACE_ALT,
            bordercolor=COLOR_SURFACE_ALT,
            arrowcolor=COLOR_MUTED,
        )

    def _create_card(self, parent, padx=20, pady=18):
        return tk.Frame(
            parent,
            bg=COLOR_SURFACE,
            padx=padx,
            pady=pady,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
        )

    def _create_button(self, parent, text, command, width, primary=False):
        if primary:
            background = COLOR_PRIMARY
            foreground = "#FFFFFF"
            active_background = COLOR_PRIMARY_HOVER
            active_foreground = "#FFFFFF"
        else:
            background = COLOR_SURFACE_ALT
            foreground = COLOR_TEXT
            active_background = COLOR_PRIMARY_SOFT
            active_foreground = COLOR_PRIMARY

        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            height=2,
            font=(FONT_FAMILY, 9, "bold"),
            bg=background,
            fg=foreground,
            activebackground=active_background,
            activeforeground=active_foreground,
            disabledforeground="#94A3B8",
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            padx=10,
        )

    def _create_stat_card(self, parent, title, value_var, accent):
        card = tk.Frame(
            parent,
            bg=COLOR_SURFACE,
            padx=16,
            pady=14,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
        )
        card.columnconfigure(0, weight=1)

        stripe = tk.Frame(card, bg=accent, height=4)
        stripe.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        tk.Label(
            card,
            text=title,
            font=(FONT_FAMILY, 8, "bold"),
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
            anchor="w",
        ).grid(row=1, column=0, sticky="w")

        tk.Label(
            card,
            textvariable=value_var,
            font=(FONT_FAMILY, 20, "bold"),
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
            anchor="w",
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))

        return card

    # ======================================================
    # Layout
    # ======================================================

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        page = tk.Frame(self.root, bg=COLOR_BG)
        page.grid(row=0, column=0, sticky="nsew")
        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)

        self._build_header(page)
        self._build_content(page)

    def _build_header(self, parent):
        header = tk.Frame(parent, bg=COLOR_HEADER, padx=32, pady=22)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        logo = tk.Frame(
            header,
            bg=COLOR_PRIMARY,
            width=54,
            height=54,
            highlightbackground="#3B82F6",
            highlightthickness=1,
        )
        logo.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 16))
        logo.grid_propagate(False)

        tk.Label(
            logo,
            text="LS",
            font=(FONT_FAMILY, 16, "bold"),
            fg="#FFFFFF",
            bg=COLOR_PRIMARY,
        ).place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(
            header,
            text="LeakShield",
            font=(FONT_FAMILY, 25, "bold"),
            fg="#FFFFFF",
            bg=COLOR_HEADER,
            anchor="w",
        ).grid(row=0, column=1, sticky="w")

        tk.Label(
            header,
            text="Hebrew PDF sensitive-information leakage detection",
            font=(FONT_FAMILY, 10),
            fg="#BFDBFE",
            bg=COLOR_HEADER,
            anchor="w",
        ).grid(row=1, column=1, sticky="w", pady=(3, 0))

        self.header_status = tk.Label(
            header,
            text="SYSTEM READY",
            font=(FONT_FAMILY, 8, "bold"),
            fg="#BBF7D0",
            bg="#123D2A",
            padx=12,
            pady=7,
        )
        self.header_status.grid(row=0, column=2, rowspan=2, sticky="e")

    def _build_content(self, parent):
        content = tk.Frame(parent, bg=COLOR_BG, padx=24, pady=20)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=3, uniform="main")
        content.columnconfigure(1, weight=2, uniform="main")
        content.rowconfigure(2, weight=1)

        # File card spans full width
        file_card = self._create_card(content, padx=20, pady=16)
        file_card.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        file_card.columnconfigure(0, weight=1)

        tk.Label(
            file_card,
            text="Document selection",
            font=(FONT_FAMILY, 12, "bold"),
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            file_card,
            text="Choose a PDF document for analysis. The selection button is locked while analysis is running.",
            font=(FONT_FAMILY, 9),
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(3, 12))

        file_row = tk.Frame(
            file_card,
            bg=COLOR_SURFACE_ALT,
            padx=12,
            pady=10,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
        )
        file_row.grid(row=2, column=0, sticky="ew")
        file_row.columnconfigure(0, weight=1)

        self.selected_file_label = tk.Label(
            file_row,
            textvariable=self.selected_pdf_path,
            font=(FONT_FAMILY, 10),
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE_ALT,
            anchor="w",
            justify="left",
            wraplength=880,
        )
        self.selected_file_label.grid(row=0, column=0, sticky="ew")

        self.choose_button = self._create_button(
            file_row,
            text="Choose PDF",
            command=self.choose_pdf,
            width=15,
            primary=False,
        )
        self.choose_button.grid(row=0, column=1, padx=(12, 0))

        # Action/progress card left
        action_card = self._create_card(content)
        action_card.grid(row=1, column=0, sticky="nsew", padx=(0, 7), pady=(0, 14))
        action_card.columnconfigure(0, weight=1)

        tk.Label(
            action_card,
            text="Analysis control",
            font=(FONT_FAMILY, 12, "bold"),
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            action_card,
            text="Run the model and generate JSON, CSV and a marked PDF report.",
            font=(FONT_FAMILY, 9),
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
        ).grid(row=1, column=0, sticky="w", pady=(3, 14))

        button_row = tk.Frame(action_card, bg=COLOR_SURFACE)
        button_row.grid(row=2, column=0, sticky="w")

        self.analyze_button = self._create_button(
            button_row,
            text="Analyze PDF",
            command=self.start_analysis,
            width=17,
            primary=True,
        )
        self.analyze_button.pack(side="left", padx=(0, 10))
        self.analyze_button.config(state="disabled")

        self.open_marked_pdf_button = self._create_button(
            button_row,
            text="Open Marked PDF",
            command=self.open_marked_pdf_report,
            width=18,
            primary=False,
        )
        self.open_marked_pdf_button.pack(side="left")
        self.open_marked_pdf_button.config(state="disabled")

        progress_header = tk.Frame(action_card, bg=COLOR_SURFACE)
        progress_header.grid(row=3, column=0, sticky="ew", pady=(18, 6))
        progress_header.columnconfigure(0, weight=1)

        tk.Label(
            progress_header,
            textvariable=self.progress_text,
            font=(FONT_FAMILY, 9),
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.progress_percent_label = tk.Label(
            progress_header,
            text="0%",
            font=(FONT_FAMILY, 9, "bold"),
            fg=COLOR_PRIMARY,
            bg=COLOR_SURFACE,
            anchor="e",
        )
        self.progress_percent_label.grid(row=0, column=1, sticky="e")

        self.progress_bar = ttk.Progressbar(
            action_card,
            mode="determinate",
            maximum=100,
            variable=self.progress_value,
            style="LeakShield.Horizontal.TProgressbar",
        )
        self.progress_bar.grid(row=4, column=0, sticky="ew")

        # Output card right
        output_card = self._create_card(content)
        output_card.grid(row=1, column=1, sticky="nsew", padx=(7, 0), pady=(0, 14))
        output_card.columnconfigure(0, weight=1)

        tk.Label(
            output_card,
            text="Generated outputs",
            font=(FONT_FAMILY, 12, "bold"),
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            output_card,
            text="Open result files directly after a successful analysis.",
            font=(FONT_FAMILY, 9),
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
        ).grid(row=1, column=0, sticky="w", pady=(3, 14))

        output_buttons = tk.Frame(output_card, bg=COLOR_SURFACE)
        output_buttons.grid(row=2, column=0, sticky="w")

        self.open_json_button = self._create_button(
            output_buttons,
            text="Open JSON",
            command=self.open_json_results,
            width=14,
            primary=False,
        )
        self.open_json_button.pack(side="left", padx=(0, 10))
        self.open_json_button.config(state="disabled")

        self.open_csv_button = self._create_button(
            output_buttons,
            text="Open CSV",
            command=self.open_csv_results,
            width=14,
            primary=False,
        )
        self.open_csv_button.pack(side="left")
        self.open_csv_button.config(state="disabled")

        status_row = tk.Frame(output_card, bg=COLOR_SURFACE)
        status_row.grid(row=3, column=0, sticky="ew", pady=(18, 0))
        status_row.columnconfigure(1, weight=1)

        self.status_badge = tk.Label(
            status_row,
            text="READY",
            font=(FONT_FAMILY, 8, "bold"),
            fg=COLOR_SUCCESS,
            bg=COLOR_SUCCESS_BG,
            padx=10,
            pady=5,
        )
        self.status_badge.grid(row=0, column=0, sticky="w", padx=(0, 10))

        tk.Label(
            status_row,
            textvariable=self.status_text,
            font=(FONT_FAMILY, 9),
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
            anchor="w",
        ).grid(row=0, column=1, sticky="ew")

        # Bottom area: summary + metrics
        summary_card = self._create_card(content, padx=18, pady=16)
        summary_card.grid(row=2, column=0, sticky="nsew", padx=(0, 7))
        summary_card.columnconfigure(0, weight=1)
        summary_card.rowconfigure(2, weight=1)

        tk.Label(
            summary_card,
            text="Analysis summary",
            font=(FONT_FAMILY, 13, "bold"),
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE,
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        tk.Label(
            summary_card,
            text="Complete analysis information, file paths and report details.",
            font=(FONT_FAMILY, 9),
            fg=COLOR_MUTED,
            bg=COLOR_SURFACE,
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(3, 12))

        text_container = tk.Frame(
            summary_card,
            bg=COLOR_SURFACE_ALT,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
        )
        text_container.grid(row=2, column=0, sticky="nsew")
        text_container.columnconfigure(0, weight=1)
        text_container.rowconfigure(0, weight=1)

        self.summary_text_box = tk.Text(
            text_container,
            wrap="word",
            font=(FONT_FAMILY, 10),
            fg=COLOR_TEXT,
            bg=COLOR_SURFACE_ALT,
            relief="flat",
            borderwidth=0,
            padx=16,
            pady=14,
            spacing1=2,
            spacing3=5,
            insertbackground=COLOR_PRIMARY,
        )
        self.summary_text_box.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(
            text_container,
            orient="vertical",
            command=self.summary_text_box.yview,
            style="LeakShield.Vertical.TScrollbar",
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.summary_text_box.configure(yscrollcommand=scrollbar.set)
        self.summary_text_box.insert(
            "1.0",
            "Select a PDF document and click Analyze PDF to begin.",
        )
        self.summary_text_box.config(state="disabled")

        metrics_area = tk.Frame(content, bg=COLOR_BG)
        metrics_area.grid(row=2, column=1, sticky="nsew", padx=(7, 0))
        metrics_area.columnconfigure(0, weight=1)
        metrics_area.columnconfigure(1, weight=1)
        metrics_area.rowconfigure(0, weight=1)
        metrics_area.rowconfigure(1, weight=1)

        self.pages_value = tk.StringVar(value="—")
        self.units_value = tk.StringVar(value="—")
        self.leaks_value = tk.StringVar(value="—")
        self.clean_value = tk.StringVar(value="—")

        self._create_stat_card(metrics_area, "PAGES", self.pages_value, COLOR_PRIMARY).grid(
            row=0, column=0, sticky="nsew", padx=(0, 7), pady=(0, 7)
        )
        self._create_stat_card(metrics_area, "TEXT UNITS", self.units_value, COLOR_ACCENT).grid(
            row=0, column=1, sticky="nsew", padx=(7, 0), pady=(0, 7)
        )
        self._create_stat_card(metrics_area, "DETECTED LEAKS", self.leaks_value, COLOR_ERROR).grid(
            row=1, column=0, sticky="nsew", padx=(0, 7), pady=(7, 0)
        )
        self._create_stat_card(metrics_area, "NON-LEAK UNITS", self.clean_value, COLOR_SUCCESS).grid(
            row=1, column=1, sticky="nsew", padx=(7, 0), pady=(7, 0)
        )

    # ======================================================
    # UI state helpers
    # ======================================================

    def _set_status_style(self, status):
        status = status.lower()

        if status == "processing":
            self.status_badge.config(
                text="PROCESSING",
                fg=COLOR_WARNING,
                bg=COLOR_WARNING_BG,
            )
            self.header_status.config(
                text="ANALYSIS IN PROGRESS",
                fg="#FDE68A",
                bg="#4A3411",
            )
        elif status == "success":
            self.status_badge.config(
                text="COMPLETED",
                fg=COLOR_SUCCESS,
                bg=COLOR_SUCCESS_BG,
            )
            self.header_status.config(
                text="ANALYSIS COMPLETED",
                fg="#BBF7D0",
                bg="#123D2A",
            )
        elif status == "error":
            self.status_badge.config(
                text="FAILED",
                fg=COLOR_ERROR,
                bg=COLOR_ERROR_BG,
            )
            self.header_status.config(
                text="ANALYSIS FAILED",
                fg="#FECACA",
                bg="#4C1D1D",
            )
        else:
            self.status_badge.config(
                text="READY",
                fg=COLOR_SUCCESS,
                bg=COLOR_SUCCESS_BG,
            )
            self.header_status.config(
                text="SYSTEM READY",
                fg="#BBF7D0",
                bg="#123D2A",
            )

    def _set_progress(self, value, text):
        value = max(0, min(100, int(value)))
        self.progress_value.set(value)
        self.progress_percent_label.config(text=f"{value}%")
        self.progress_text.set(text)
        self.root.update_idletasks()

    def _set_analysis_state(self, analyzing):
        self.is_analyzing = analyzing

        if analyzing:
            self.choose_button.config(state="disabled", cursor="arrow")
            self.analyze_button.config(state="disabled", cursor="arrow")
            self.open_json_button.config(state="disabled", cursor="arrow")
            self.open_csv_button.config(state="disabled", cursor="arrow")
            self.open_marked_pdf_button.config(state="disabled", cursor="arrow")
            self._set_status_style("processing")
        else:
            self.choose_button.config(state="normal", cursor="hand2")

    def _reset_metrics(self):
        self.pages_value.set("—")
        self.units_value.set("—")
        self.leaks_value.set("—")
        self.clean_value.set("—")

    def set_summary_text(self, text):
        self.summary_text_box.config(state="normal")
        self.summary_text_box.delete("1.0", "end")
        self.summary_text_box.insert("1.0", text)
        self.summary_text_box.see("1.0")
        self.summary_text_box.config(state="disabled")

    # ======================================================
    # Actions
    # ======================================================

    def choose_pdf(self):
        if self.is_analyzing:
            return

        file_path = filedialog.askopenfilename(
            title="Choose PDF file",
            filetypes=[("PDF files", "*.pdf")],
        )

        if not file_path:
            return

        self.selected_pdf_path.set(file_path)
        self.selected_file_label.config(fg=COLOR_TEXT)
        self.status_text.set("PDF selected. Ready to analyze.")
        self._set_status_style("ready")
        self._set_progress(0, "Document selected")
        self._reset_metrics()

        self.set_summary_text(
            "DOCUMENT READY\n"
            "────────────────────────────────────────\n\n"
            f"Selected file:\n{file_path}\n\n"
            "Click Analyze PDF to start the sensitive-information scan."
        )

        self.analyze_button.config(state="normal", cursor="hand2")
        self.open_json_button.config(state="disabled", cursor="arrow")
        self.open_csv_button.config(state="disabled", cursor="arrow")
        self.open_marked_pdf_button.config(state="disabled", cursor="arrow")

    def start_analysis(self):
        pdf_path = self.selected_pdf_path.get()

        if not pdf_path or pdf_path == "No PDF selected":
            messagebox.showwarning(
                "No file selected",
                "Please choose a PDF file first.",
            )
            return

        if not os.path.exists(pdf_path):
            messagebox.showerror(
                "File not found",
                "The selected PDF file does not exist.",
            )
            return

        if not pdf_path.lower().endswith(".pdf"):
            messagebox.showerror(
                "Invalid file",
                "Please choose a PDF file.",
            )
            return

        self._set_analysis_state(True)
        self._set_progress(5, "Preparing document analysis")
        self.status_text.set("Preparing PDF analysis...")
        self.set_summary_text(
            "ANALYSIS IN PROGRESS\n"
            "────────────────────────────────────────\n\n"
            "LeakShield is loading and processing the selected PDF.\n\n"
            "The Choose PDF button is locked until the analysis is complete."
        )

        thread = threading.Thread(
            target=self.run_analysis,
            args=(pdf_path,),
            daemon=True,
        )
        thread.start()

    def run_analysis(self, pdf_path):
        try:
            self.root.after(
                0,
                lambda: (
                    self._set_progress(15, "Extracting and normalizing PDF text"),
                    self.status_text.set("Extracting and normalizing PDF text..."),
                ),
            )

            summary = analyze_pdf(
                pdf_path=pdf_path,
                output_json=OUTPUT_JSON,
                output_csv=OUTPUT_CSV,
            )

            self.root.after(
                0,
                lambda: (
                    self._set_progress(78, "Model analysis completed"),
                    self.status_text.set("Generating marked PDF report..."),
                ),
            )

            marked_pdf_summary = create_colored_pdf_report(
                input_pdf_path=pdf_path,
                results_json_path=OUTPUT_JSON,
                output_pdf_path=OUTPUT_MARKED_PDF,
            )

            self.root.after(
                0,
                lambda: self.on_analysis_success(summary, marked_pdf_summary),
            )

        except Exception as error:
            self.root.after(
                0,
                lambda error=error: self.on_analysis_error(error),
            )

    def on_analysis_success(self, summary, marked_pdf_summary=None):
        self._set_progress(100, "Analysis and report generation completed")
        self._set_analysis_state(False)
        self._set_status_style("success")
        self.status_text.set("Analysis completed successfully.")

        total_units = int(summary.get("total_sentences", 0))
        total_leaks = int(summary.get("total_leaks", 0))
        total_non_leaks = int(
            summary.get("total_non_leaks", max(0, total_units - total_leaks))
        )

        self.pages_value.set(str(summary.get("pages", 0)))
        self.units_value.set(str(total_units))
        self.leaks_value.set(str(total_leaks))
        self.clean_value.set(str(total_non_leaks))

        marked_pdf_text = ""

        if marked_pdf_summary:
            marked_pdf_text = (
                "\n\nMARKED PDF REPORT\n"
                f"{OUTPUT_MARKED_PDF}\n"
                f"Marked leak spans: {marked_pdf_summary.get('marked_count')}\n"
                f"Unmarked leak texts: {marked_pdf_summary.get('not_found_count')}"
            )

        self.set_summary_text(
            "ANALYSIS COMPLETED SUCCESSFULLY\n"
            "────────────────────────────────────────\n\n"
            f"PDF DOCUMENT\n{summary['pdf_path']}\n\n"
            "DOCUMENT STATISTICS\n"
            f"Pages analyzed: {summary['pages']}\n"
            f"Total text units: {total_units}\n"
            f"Detected leaks: {total_leaks}\n"
            f"Non-leak units: {total_non_leaks}\n\n"
            f"JSON OUTPUT\n{summary['output_json']}\n\n"
            f"CSV OUTPUT\n{summary['output_csv']}"
            f"{marked_pdf_text}"
        )

        self.analyze_button.config(state="normal", cursor="hand2")
        self.open_json_button.config(state="normal", cursor="hand2")
        self.open_csv_button.config(state="normal", cursor="hand2")

        if os.path.exists(os.path.abspath(OUTPUT_MARKED_PDF)):
            self.open_marked_pdf_button.config(
                state="normal",
                cursor="hand2",
            )
        else:
            self.open_marked_pdf_button.config(
                state="disabled",
                cursor="arrow",
            )

        messagebox.showinfo(
            "Analysis completed",
            "PDF analysis completed successfully.\n\n"
            "You can now open the marked PDF, JSON, or CSV result files.",
        )

    def on_analysis_error(self, error):
        self._set_progress(0, "Analysis stopped")
        self._set_analysis_state(False)
        self._set_status_style("error")
        self.status_text.set("Analysis failed.")
        self._reset_metrics()

        self.set_summary_text(
            "ANALYSIS FAILED\n"
            "────────────────────────────────────────\n\n"
            f"Error:\n{error}"
        )

        self.analyze_button.config(state="normal", cursor="hand2")
        self.open_json_button.config(state="disabled", cursor="arrow")
        self.open_csv_button.config(state="disabled", cursor="arrow")
        self.open_marked_pdf_button.config(state="disabled", cursor="arrow")

        messagebox.showerror(
            "Analysis failed",
            f"An error occurred during analysis:\n{error}",
        )

    # ======================================================
    # Open generated files
    # ======================================================

    def open_json_results(self):
        json_path = os.path.abspath(OUTPUT_JSON)

        if not os.path.exists(json_path):
            messagebox.showwarning(
                "File not found",
                "results.json was not found. Please run analysis first.",
            )
            return

        os.startfile(json_path)

    def open_csv_results(self):
        csv_path = os.path.abspath(OUTPUT_CSV)

        if not os.path.exists(csv_path):
            messagebox.showwarning(
                "File not found",
                "results.csv was not found. Please run analysis first.",
            )
            return

        os.startfile(csv_path)

    def open_marked_pdf_report(self):
        marked_pdf_path = os.path.abspath(OUTPUT_MARKED_PDF)

        if not os.path.exists(marked_pdf_path):
            messagebox.showwarning(
                "File not found",
                "marked_leaks_report.pdf was not found. Please run analysis first.",
            )
            return

        os.startfile(marked_pdf_path)


def main():
    root = tk.Tk()
    LeakShieldGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
