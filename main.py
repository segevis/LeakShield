# ==========================================================
# Main Application Module
# ----------------------------------------------------------
# Command-line entry point for the system.
#
# The reusable analysis logic is implemented in:
# analysis_pipeline.analyze_pdf()
# ==========================================================

from config import PDF_PATH, OUTPUT_JSON, OUTPUT_CSV
from analysis_pipeline import analyze_pdf


def main():
    try:
        summary = analyze_pdf(pdf_path=PDF_PATH,output_json=OUTPUT_JSON,output_csv=OUTPUT_CSV,)

        print("✔ Analysis completed successfully")
        print("✔ Results saved in output folder")
        print()
        print("Summary:")
        print(f"Pages analyzed: {summary['pages']}")
        print(f"Total text units: {summary['total_sentences']}")
        print(f"Detected leaks: {summary['total_leaks']}")
        print(f"JSON output: {summary['output_json']}")
        print(f"CSV output: {summary['output_csv']}")

    except Exception as error:
        print("✖ Analysis failed")
        print(f"Error: {error}")


if __name__ == "__main__":
    main()