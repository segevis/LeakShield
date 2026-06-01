# ==========================================================
# LeakShield GUI
# ----------------------------------------------------------
# Desktop interface for analyzing PDF files.
#
# Features:
# - Choose a PDF file
# - Run analysis
# - Save JSON/CSV output
# - Show scrollable analysis summary
# - Open results.json directly after analysis
# - Open results.csv directly after analysis
# - Resizable/dynamic window layout
# ==========================================================

import os
import threading
import tkinter as tk

from tkinter import filedialog, messagebox
from analysis_pipeline import analyze_pdf


APP_TITLE = "LeakShield - Sensitive Information Leakage Detection"
OUTPUT_JSON = "output/results.json"
OUTPUT_CSV = "output/results.csv"


class LeakShieldGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("900x620")
        self.root.minsize(760, 520)
        self.root.resizable(True, True)

        self.selected_pdf_path = tk.StringVar(value="No PDF selected")
        self.status_text = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main_frame = tk.Frame(self.root, padx=24, pady=24)
        main_frame.grid(row=0, column=0, sticky="nsew")

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)

        title_label = tk.Label(
            main_frame,
            text="LeakShield",
            font=("Segoe UI", 24, "bold"),
        )
        title_label.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        subtitle_label = tk.Label(
            main_frame,
            text="Sensitive Information Leakage Detection from Hebrew PDF Documents",
            font=("Segoe UI", 11),
            fg="#555555",
        )
        subtitle_label.grid(row=1, column=0, sticky="ew", pady=(0, 24))

        file_frame = tk.LabelFrame(
            main_frame,
            text="PDF File",
            padx=16,
            pady=16,
            font=("Segoe UI", 10, "bold"),
        )
        file_frame.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        file_frame.columnconfigure(0, weight=1)

        selected_file_label = tk.Label(
            file_frame,
            textvariable=self.selected_pdf_path,
            anchor="w",
            justify="left",
            wraplength=820,
            font=("Segoe UI", 10),
        )
        selected_file_label.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        choose_button = tk.Button(
            file_frame,
            text="Choose PDF",
            width=18,
            command=self.choose_pdf,
        )
        choose_button.grid(row=1, column=0, sticky="w")

        actions_frame = tk.Frame(main_frame)
        actions_frame.grid(row=3, column=0, sticky="ew", pady=(0, 16))

        self.analyze_button = tk.Button(
            actions_frame,
            text="Analyze PDF",
            width=18,
            height=2,
            command=self.start_analysis,
            state="disabled",
        )
        self.analyze_button.pack(side="left", padx=(0, 12))

        self.open_json_button = tk.Button(
            actions_frame,
            text="Open JSON",
            width=14,
            height=2,
            command=self.open_json_results,
            state="disabled",
        )
        self.open_json_button.pack(side="left", padx=(0, 12))

        self.open_csv_button = tk.Button(
            actions_frame,
            text="Open CSV",
            width=14,
            height=2,
            command=self.open_csv_results,
            state="disabled",
        )
        self.open_csv_button.pack(side="left")

        summary_frame = tk.LabelFrame(
            main_frame,
            text="Analysis Summary",
            padx=12,
            pady=12,
            font=("Segoe UI", 10, "bold"),
        )
        summary_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 16))

        summary_frame.columnconfigure(0, weight=1)
        summary_frame.rowconfigure(0, weight=1)

        self.summary_text_box = tk.Text(
            summary_frame,
            wrap="word",
            font=("Segoe UI", 10),
            height=12,
            relief="flat",
            padx=10,
            pady=10,
        )
        self.summary_text_box.grid(row=0, column=0, sticky="nsew")

        summary_scrollbar = tk.Scrollbar(
            summary_frame,
            orient="vertical",
            command=self.summary_text_box.yview,
        )
        summary_scrollbar.grid(row=0, column=1, sticky="ns")

        self.summary_text_box.configure(yscrollcommand=summary_scrollbar.set)
        self.summary_text_box.insert("1.0", "Select a PDF file and click Analyze.")
        self.summary_text_box.config(state="disabled")

        status_frame = tk.LabelFrame(
            main_frame,
            text="Status",
            padx=16,
            pady=12,
            font=("Segoe UI", 10, "bold"),
        )
        status_frame.grid(row=5, column=0, sticky="ew", pady=(0, 12))
        status_frame.columnconfigure(0, weight=1)

        status_label = tk.Label(
            status_frame,
            textvariable=self.status_text,
            anchor="w",
            font=("Segoe UI", 10),
        )
        status_label.grid(row=0, column=0, sticky="ew")

        footer_label = tk.Label(
            main_frame,
            text="After analysis, you can open the generated JSON and CSV result files directly.",
            font=("Segoe UI", 9),
            fg="#777777",
        )
        footer_label.grid(row=6, column=0, sticky="ew")

    def set_summary_text(self, text):
        self.summary_text_box.config(state="normal")
        self.summary_text_box.delete("1.0", "end")
        self.summary_text_box.insert("1.0", text)
        self.summary_text_box.config(state="disabled")

    def choose_pdf(self):
        file_path = filedialog.askopenfilename(
            title="Choose PDF file",
            filetypes=[("PDF files", "*.pdf")],
        )

        if not file_path:
            return

        self.selected_pdf_path.set(file_path)
        self.status_text.set("PDF selected. Ready to analyze.")
        self.set_summary_text("Click Analyze PDF to start processing.")
        self.analyze_button.config(state="normal")

        self.open_json_button.config(state="disabled")
        self.open_csv_button.config(state="disabled")

    def start_analysis(self):
        pdf_path = self.selected_pdf_path.get()

        if not pdf_path or pdf_path == "No PDF selected":
            messagebox.showwarning("No file selected", "Please choose a PDF file first.")
            return

        if not os.path.exists(pdf_path):
            messagebox.showerror("File not found", "The selected PDF file does not exist.")
            return

        if not pdf_path.lower().endswith(".pdf"):
            messagebox.showerror("Invalid file", "Please choose a PDF file.")
            return

        self.analyze_button.config(state="disabled")
        self.open_json_button.config(state="disabled")
        self.open_csv_button.config(state="disabled")

        self.status_text.set("Processing PDF... This may take a moment.")
        self.set_summary_text("Analyzing document. Please wait...")

        thread = threading.Thread(
            target=self.run_analysis,
            args=(pdf_path,),
            daemon=True,
        )
        thread.start()

    def run_analysis(self, pdf_path):
        try:
            summary = analyze_pdf(
                pdf_path=pdf_path,
                output_json=OUTPUT_JSON,
                output_csv=OUTPUT_CSV,
            )

            self.root.after(0, lambda: self.on_analysis_success(summary))

        except Exception as error:
            self.root.after(0, lambda: self.on_analysis_error(error))

    def on_analysis_success(self, summary):
        self.status_text.set("Analysis completed successfully.")

        self.set_summary_text(
            "Analysis completed successfully.\n\n"
            f"PDF file:\n{summary['pdf_path']}\n\n"
            f"Pages analyzed: {summary['pages']}\n"
            f"Total text units: {summary['total_sentences']}\n"
            f"Detected leaks: {summary['total_leaks']}\n\n"
            f"JSON output:\n{summary['output_json']}\n\n"
            f"CSV output:\n{summary['output_csv']}"
        )

        self.analyze_button.config(state="normal")
        self.open_json_button.config(state="normal")
        self.open_csv_button.config(state="normal")

        messagebox.showinfo(
            "Analysis completed",
            "PDF analysis completed successfully.\nYou can now open the JSON or CSV result files.",
        )

    def on_analysis_error(self, error):
        self.status_text.set("Analysis failed.")
        self.set_summary_text(f"Error:\n{error}")

        self.analyze_button.config(state="normal")
        self.open_json_button.config(state="disabled")
        self.open_csv_button.config(state="disabled")

        messagebox.showerror(
            "Analysis failed",
            f"An error occurred during analysis:\n{error}",
        )

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


def main():
    root = tk.Tk()
    app = LeakShieldGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()