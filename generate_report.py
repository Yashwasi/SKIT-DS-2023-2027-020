import subprocess
import sys
import io
import html
from collections import defaultdict
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether
)

ROOT = Path(__file__).resolve().parent
REPORT_DIR = ROOT / "weekly_reports"
COLLEGE_NAME = "Swami Keshvanand Institute of Technology, Management & Gramothan, Jaipur"
DEPARTMENT_NAME = "Department of Computer Science & Engineering"
TZ = ZoneInfo("Asia/Kolkata")


def git_output(*args):
    result = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=True
    )
    return result.stdout.strip()


def reporting_window(period, now):
    """Weekly window is Friday 00:00 to the following Friday 00:00 IST."""
    today = now.date()
    if period == "weekly":
        days_since_friday = (today.weekday() - 4) % 7 + (7 if today.weekday() != 4 else 0)
        start_day = today - timedelta(days=days_since_friday)
        # On Friday, generate the previous completed Friday-Thursday period.
        if days_since_friday <= 1:
            start_day -= timedelta(days=7)
        end_day = start_day + timedelta(days=7)
        return start_day, end_day, f"{start_day:%d %b %Y} - {(end_day - timedelta(days=1)):%d %b %Y}"
    if period == "monthly":
        start_day = today - timedelta(days=30)
        return start_day, today + timedelta(days=1), f"Last 30 days from {start_day:%d %b %Y}"
    return date(2000, 1, 1), today + timedelta(days=1), "Complete project history"


def get_repo_info():
    try:
        repo_name = Path(git_output("rev-parse", "--show-toplevel")).name
    except Exception:
        repo_name = ROOT.name
    try:
        branch_name = git_output("rev-parse", "--abbrev-ref", "HEAD")
    except Exception:
        branch_name = "unknown"
    return repo_name, branch_name


def get_git_metrics(period, start_day, end_day):
    # Date bounds are explicit; --no-merges avoids counting merge commits as individual work.
    raw = git_output(
        "log", "main", "--no-merges",
        "--pretty=format:COMMIT|||%h|||%an|||%ad|||%s",
        "--date=short", "--numstat",
        f"--since={start_day.isoformat()} 00:00:00",
        f"--before={end_day.isoformat()} 00:00:00",
    )
    students = defaultdict(lambda: {
        "commits": 0, "added": 0, "deleted": 0, "active_days": set()
    })
    timeline = defaultdict(lambda: defaultdict(int))
    logs = defaultdict(list)
    current_author = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("COMMIT|||"):
            parts = line.split("|||", 4)
            if len(parts) < 5:
                current_author = None
                continue
            sha, author, day, message = parts[1:]
            if "bot" in author.lower() or "github-actions" in author.lower():
                current_author = None
                continue
            current_author = author
            students[author]["commits"] += 1
            students[author]["active_days"].add(day)
            logs[author].append((day, sha, message))
            try:
                dt = datetime.strptime(day, "%Y-%m-%d").date()
                key = dt.strftime("%a %d %b") if period == "weekly" else (
                    f"{dt.isocalendar().year}-W{dt.isocalendar().week:02d}"
                    if period == "monthly" else dt.strftime("%Y-%m")
                )
                timeline[key][author] += 1
            except ValueError:
                pass
        elif current_author:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
                students[current_author]["added"] += int(parts[0])
                students[current_author]["deleted"] += int(parts[1])
    return students, timeline, logs


def create_charts(students, timeline):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.0))
    authors = list(students)
    periods = list(timeline)
    if periods and authors:
        for author in authors:
            ax1.plot(periods, [timeline[p].get(author, 0) for p in periods],
                     marker="o", linewidth=1.5, label=author)
        ax1.set_title("Commit Timeline", fontsize=8)
        ax1.set_ylabel("Commits", fontsize=7)
        ax1.tick_params(axis="x", rotation=30, labelsize=6)
        ax1.tick_params(axis="y", labelsize=6)
        ax1.grid(True, linestyle="--", alpha=0.4)
        ax1.legend(fontsize=5.5)
    else:
        ax1.text(0.5, 0.5, "No commits in this period", ha="center", va="center")
        ax1.set_axis_off()

    if authors:
        net = [students[a]["added"] - students[a]["deleted"] for a in authors]
        ax2.bar(authors, net)
        ax2.set_title("Net Lines Changed", fontsize=8)
        ax2.set_ylabel("Added - Deleted", fontsize=7)
        ax2.tick_params(axis="x", rotation=30, labelsize=6)
        ax2.tick_params(axis="y", labelsize=6)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
    else:
        ax2.text(0.5, 0.5, "No code changes recorded", ha="center", va="center")
        ax2.set_axis_off()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image(buf, width=520, height=95)


def generate_report(period="weekly"):
    now = datetime.now(TZ)
    start_day, end_day, window_label = reporting_window(period, now)
    repo_name, branch_name = get_repo_info()
    students, timeline, logs = get_git_metrics(period, start_day, end_day)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    output = REPORT_DIR / f"weekly_report_{stamp}.pdf" if period == "weekly" else REPORT_DIR / f"{period}_report_{stamp}.pdf"

    doc = SimpleDocTemplate(str(output), pagesize=letter,
                            rightMargin=24, leftMargin=24, topMargin=16, bottomMargin=16)
    styles = getSampleStyleSheet()
    college_style = ParagraphStyle("College", parent=styles["Heading1"], fontSize=11,
                                   leading=13, alignment=1, spaceAfter=1)
    dept_style = ParagraphStyle("Dept", parent=styles["Normal"], fontSize=8,
                                alignment=1, spaceAfter=2)
    title_style = ParagraphStyle("FormTitle", parent=styles["Heading2"], fontSize=11,
                                 alignment=1, spaceAfter=3)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=8.5,
                                   leading=10, spaceBefore=3, spaceAfter=2)
    small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=6.5, leading=8)
    cell = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=6.5, leading=8)

    story = [
        Paragraph(f"<b>{html.escape(COLLEGE_NAME)}</b>", college_style),
        Paragraph(f"<b>{html.escape(DEPARTMENT_NAME)}</b>", dept_style),
        Paragraph(f"<u><b>{'Weekly Progress Report (Form-3)' if period == 'weekly' else period.title() + ' Project Report'}</b></u>", title_style),
        Paragraph(f"<b>Repository:</b> {html.escape(repo_name)} &nbsp; | &nbsp; <b>Branch:</b> {html.escape(branch_name)}", small),
        Paragraph(f"<b>Evaluation Window:</b> {html.escape(window_label)} &nbsp; | &nbsp; <b>Generated:</b> {now:%d %b %Y, %H:%M IST}", small),
        Spacer(1, 3),
        Paragraph("1. Individual Contribution Breakdown", section_style),
    ]
    total_commits = sum(v["commits"] for v in students.values())
    data = [["Student Name", "Commits (%)", "Lines Added", "Lines Deleted", "Net LOC", "Active Days"]]
    for name, v in sorted(students.items()):
        pct = v["commits"] * 100 / total_commits if total_commits else 0
        data.append([Paragraph(html.escape(name), cell), f"{v['commits']} ({pct:.1f}%)",
                     f"+{v['added']:,}", f"-{v['deleted']:,}",
                     f"{v['added'] - v['deleted']:,}", str(len(v["active_days"]))])
    if not students:
        data.append(["No commits found in this reporting period.", "-", "-", "-", "-", "-"])
    table = Table(data, colWidths=[130, 75, 75, 75, 65, 65], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    story.extend([table, Spacer(1, 3), Paragraph("2. Visual Trends & Volume", section_style)])
    story.append(create_charts(students, timeline))
    story.extend([Spacer(1, 2), Paragraph(f"3. Detailed Commit Logs & Mentor Evaluation ({period.title()})", section_style)])

    if not logs:
        story.append(Paragraph("No commit logs found for this reporting period.", small))
    else:
        for name, entries in sorted(logs.items()):
            block = [Paragraph(f"<b>Student:</b> {html.escape(name)} - {len(entries)} commit(s)", small)]
            log_data = [[Paragraph("<b>Date</b>", cell), Paragraph("<b>Hash</b>", cell),
                         Paragraph("<b>Commit Message</b>", cell), Paragraph("<b>Mentor Marks (/10)</b>", cell)]]
            for idx, (day, sha, msg) in enumerate(entries):
                log_data.append([Paragraph(html.escape(day), cell), Paragraph(html.escape(sha), cell),
                                 Paragraph(html.escape(msg or "(No commit message)"), cell),
                                 Paragraph("<b>_____ / 10</b>" if idx == 0 else "", cell)])
            lt = Table(log_data, colWidths=[65, 48, 332, 90], repeatRows=1)
            lt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#475569")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONTSIZE", (0, 0), (-1, -1), 6.5),
                ("TOPPADDING", (0, 0), (-1, -1), 1.2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.2),
                ("SPAN", (3, 1), (3, len(log_data) - 1)) if len(log_data) > 2 else ("BACKGROUND", (3, 1), (3, 1), colors.HexColor("#FEF3C7")),
            ]))
            block.extend([lt, Spacer(1, 2)])
            story.append(KeepTogether(block))

    story.extend([
        Spacer(1, 4),
        Table([[
            [Paragraph("<b>Name:</b> ____________________", small),
             Paragraph("<b>Designation:</b> Project Mentor", small),
             Spacer(1, 3), Paragraph("<b>Signature:</b> ________________", small)],
            [Paragraph("<b>Name:</b> ____________________", small),
             Paragraph("<b>Designation:</b> Lab Coordinator", small),
             Spacer(1, 3), Paragraph("<b>Signature:</b> ________________", small)]
        ]], colWidths=[270, 270], style=TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
        ])),
        Spacer(1, 3),
        Paragraph("<i>Note: Git metrics summarize recorded repository activity; they do not alone measure contribution quality. Review names, dates, and mentor marks before submission.</i>", small)
    ])
    doc.build(story)
    print(f"[SUCCESS] Generated: {output}")
    print(f"Students: {len(students)} | Commits: {total_commits} | Window: {window_label}")
    return output


if __name__ == "__main__":
    selected = sys.argv[1].lower() if len(sys.argv) > 1 else "weekly"
    if selected not in {"weekly", "monthly", "final"}:
        raise SystemExit("Usage: python generate_report.py [weekly|monthly|final]")
    generate_report(selected)



