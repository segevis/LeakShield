# ==========================================================
# Main Entry Point
# ----------------------------------------------------------
# Default behavior:
#   python main.py
#   -> opens the GUI
#
# CLI behavior:
#   python main.py --cli
#   -> runs the analysis directly from the terminal
#
# After CLI analysis:
#   - results.json is created
#   - results.csv is created
#   - marked_leaks_report.pdf is created
# ==========================================================

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

from analysis_pipeline import analyze_pdf
from config import OUTPUT_CSV, OUTPUT_JSON, OUTPUT_MARKED_PDF, PDF_PATH
from pdf_colored_report import create_colored_pdf_report


def run_cli_analysis() -> None:
    """
    Run the PDF leakage analysis directly from the terminal.
    Uses the production Regex + HeBERT Multi-Task V2 pipeline.
    """
    results = analyze_pdf(PDF_PATH)

    output_json = Path(OUTPUT_JSON)
    output_csv = Path(OUTPUT_CSV)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # analysis_pipeline may already save results internally.
    # This block is defensive: if results are returned but JSON was not saved,
    # save the JSON here.
    if results is not None and not output_json.exists():
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Create marked PDF report after results.json exists.
    try:
        report_summary = create_colored_pdf_report(
            input_pdf_path=PDF_PATH,
            results_json_path=OUTPUT_JSON,
            output_pdf_path=OUTPUT_MARKED_PDF,
        )
        print(f"✔ Marked PDF report saved: {OUTPUT_MARKED_PDF}")
        print(f"Marked leak spans: {report_summary.get('marked_count')}")
    except Exception as exc:
        print(f"⚠ Could not create marked PDF report: {exc}")

    print("✔ Analysis completed successfully")
    print("✔ Results saved in output folder")
    print()
    print("Summary:")

    try:
        with output_json.open("r", encoding="utf-8") as f:
            saved_results = json.load(f)

        summary = saved_results.get("summary", {})
        print(f"Pages analyzed: {summary.get('total_pages', 0)}")
        print(f"Total sentences: {summary.get('total_sentences', 0)}")
        print(f"Detected leaks: {summary.get('total_leaks', 0)}")
        print(f"Non-leaks: {summary.get('total_non_leaks', 0)}")
        print(f"Undetermined: {summary.get('total_undetermined', 0)}")

    except Exception:
        # If the JSON structure is different, do not fail the CLI.
        pass

    print(f"JSON output: {OUTPUT_JSON}")
    print(f"CSV output: {OUTPUT_CSV}")
    print(f"Marked PDF output: {OUTPUT_MARKED_PDF}")


def run_gui() -> None:
    """
    Run gui.py as the application entry point.

    This avoids depending on a specific function name inside gui.py.
    If gui.py already works when running:
        python gui.py
    then this will work too.
    """
    gui_path = Path("gui.py")

    if not gui_path.exists():
        raise FileNotFoundError("gui.py was not found in the project root.")

    runpy.run_path(str(gui_path), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LeakShield - Regex and HeBERT Multi-Task PDF Leakage Detection"
    )

    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run direct terminal analysis instead of opening the GUI",
    )

    args = parser.parse_args()

    if args.cli:
        run_cli_analysis()
    else:
        run_gui()


if __name__ == "__main__":
    main()