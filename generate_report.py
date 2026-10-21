
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "weekly_reports"


def git_output(*args):
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def generate_report(period="weekly"):
    now = datetime.now()
    if period == "weekly":
        since = now - timedelta(days=7)
    elif period == "monthly":
        since = now - timedelta(days=30)
    elif period == "final":
        since = None
    else:
        raise ValueError("Use weekly, monthly, or final")

    command = [
        "log", "--all",
        "--date=short",
        "--pretty=format:%ad|%an|%h|%s",
    ]
    if since:
        command.insert(1, f"--since={since.strftime('%Y-%m-%d %H:%M:%S')}")

    output = git_output(*command)
    commits = [line.split("|", 3) for line in output.splitlines() if line]

    REPORT_DIR.mkdir(exist_ok=True)
    filename = REPORT_DIR / f"{period}_report_{now:%Y%m%d_%H%M%S}.pdf"

    styles = getSampleStyleSheet()
    story = [
        Paragraph("Rain Drop Volume Prediction Using AI", styles["Title"]),
        Paragraph(f"{period.title()} GitHub Contribution Report", styles["Heading2"]),
        Paragraph(f"Generated: {now:%d %B %Y, %H:%M}", styles["Normal"]),
        Spacer(1, 0.2 * inch),
        Paragraph(f"Commits found: {len(commits)}", styles["Heading3"]),
        Spacer(1, 0.15 * inch),
    ]

    rows = [["Date", "Contributor", "Commit", "Message"]]
    for date, author, commit_hash, message in commits:
        rows.append([
            date,
            Paragraph(author, styles["BodyText"]),
            commit_hash,
            Paragraph(message, styles["BodyText"]),
        ])

    if len(rows) == 1:
        rows.append(["—", "No commits", "—", "No commits found in this period"])

    table = Table(
        rows,
        repeatRows=1,
        colWidths=[0.85 * inch, 1.35 * inch, 0.8 * inch, 3.1 * inch],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#23466D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F1F5F9")]),
    ]))

    story.append(table)
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Note: This report reflects Git commit history available locally. "
        "It does not measure uncommitted work or prove the quality of contributions.",
        styles["Italic"],
    ))

    SimpleDocTemplate(
        str(filename),
        pagesize=A4,
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
    ).build(story)

    print(f"Report created: {filename}")


if __name__ == "__main__":
    period = sys.argv[1].lower() if len(sys.argv) > 1 else "weekly"
    generate_report(period)
