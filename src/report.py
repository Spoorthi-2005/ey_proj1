import os
import pandas as pd
from fpdf import FPDF

from .config import OUTPUT_DIR


def build_weekly_report() -> str:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "IntelliReview Weekly Report", ln=True)

    def _safe_text(x: object, max_len: int = 120) -> str:
        s = str(x) if x is not None else ""
        s = s.replace("\n", " ")
        return (s[: max_len - 1] + "…") if len(s) > max_len else s

    def add_table(title: str, df: pd.DataFrame, max_rows=15, max_cols=6):
        # Reduce columns to fit width
        if df.shape[1] > max_cols:
            df = df.iloc[:, :max_cols]
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, title, ln=True)
        pdf.set_font("Arial", size=9)
        epw = pdf.w - 2 * pdf.l_margin  # effective page width

        # Header
        header = " | ".join([_safe_text(c, 40) for c in df.columns])
        pdf.multi_cell(w=epw, h=6, txt=header, new_x="LMARGIN", new_y="NEXT")

        # Rows
        for _, row in df.head(max_rows).iterrows():
            line = " | ".join([_safe_text(x) for x in row.to_list()])
            pdf.multi_cell(w=epw, h=6, txt=line, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    # Attachments if exist
    for name in ["kpi_impact.parquet", "segments.parquet", "fake_reviewers.parquet", "alerts.parquet"]:
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            add_table(name.replace(".parquet", "").title(), df)

    out_path = os.path.join(OUTPUT_DIR, "weekly_report.pdf")
    pdf.output(out_path)
    return out_path

if __name__ == "__main__":
    print(build_weekly_report())
