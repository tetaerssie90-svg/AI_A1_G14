#!/usr/bin/env python3
"""Build AI_A1_G14 PDFs in the same INES assignment document format."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
LOGO = Path(__file__).resolve().parent / "ines_logo.png"
W, H = A4

HEADER_BG = HexColor("#DCE4D5")
NAVY = HexColor("#17365D")
ROW_ALT = HexColor("#F4F7FA")
LINE = HexColor("#8EA3B8")
SCREEN_BG = HexColor("#F7F9FB")
WARN_BG = HexColor("#FFF4E5")


def styles():
    base = getSampleStyleSheet()
    return {
        "course": ParagraphStyle(
            "course",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=15,
            textColor=black,
            spaceAfter=0,
        ),
        "title": ParagraphStyle(
            "title",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=black,
            spaceBefore=2,
            spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=10,
            leading=13,
            textColor=black,
            spaceAfter=8,
        ),
        "heading": ParagraphStyle(
            "heading",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "th": ParagraphStyle(
            "th",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=white,
        ),
        "td": ParagraphStyle(
            "td",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=black,
        ),
        "tdbold": ParagraphStyle(
            "tdbold",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=black,
        ),
        "screen_title": ParagraphStyle(
            "screen_title",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=white,
        ),
        "screen_body": ParagraphStyle(
            "screen_body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=black,
        ),
        "screen_bold": ParagraphStyle(
            "screen_bold",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=black,
        ),
    }


S = styles()


def P(text: str, style="td"):
    return Paragraph(str(text), S[style])


def navy_table(rows: list[list], col_widths: list[float]) -> Table:
    data = []
    for r, row in enumerate(rows):
        styled = []
        for i, cell in enumerate(row):
            if isinstance(cell, Paragraph):
                styled.append(cell)
            elif r == 0:
                styled.append(P(cell, "th"))
            elif i == 0:
                styled.append(P(cell, "tdbold"))
            else:
                styled.append(P(cell, "td"))
        data.append(styled)
    table = Table(data, colWidths=col_widths, repeatRows=1)
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, 0), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("BACKGROUND", (0, 1), (-1, -1), white),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            commands.append(("BACKGROUND", (0, i), (-1, i), ROW_ALT))
    table.setStyle(TableStyle(commands))
    return table


def screen_frame(title: str, inner_rows: list[list], col_widths: list[float]) -> Table:
    inner = []
    for row in inner_rows:
        inner.append([
            cell if isinstance(cell, Paragraph) else P(cell, "screen_body")
            for cell in row
        ])
    body = Table(inner, colWidths=col_widths)
    body.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SCREEN_BG),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("BOX", (0, 0), (-1, -1), 0.3, LINE),
                ("LINEBELOW", (0, 0), (-1, -2), 0.2, LINE),
            ]
        )
    )
    header = Table([[P(title, "screen_title")]], colWidths=[sum(col_widths)])
    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    frame = Table([[header], [body]], colWidths=[sum(col_widths)])
    frame.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.8, NAVY),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    return frame


def draw_header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(HEADER_BG)
    canvas.rect(0, H - 30 * mm, W, 30 * mm, fill=1, stroke=0)
    if LOGO.exists():
        canvas.drawImage(
            str(LOGO),
            12 * mm,
            H - 28 * mm,
            width=20 * mm,
            height=20 * mm,
            preserveAspectRatio=True,
            mask="auto",
        )
    canvas.setFillColor(black)
    canvas.setFont("Times-Bold", 11)
    canvas.drawCentredString(W / 2 + 6 * mm, H - 11 * mm, "INSTITUT D'ENSEIGNEMENT SUPÉRIEUR DE RUHENGERI")
    canvas.setFont("Times-Roman", 8)
    canvas.drawCentredString(W / 2 + 6 * mm, H - 16.5 * mm, "B.P. 155, Ruhengeri, Rwanda")
    canvas.setFont("Times-Roman", 7.5)
    canvas.drawCentredString(
        W / 2 + 6 * mm,
        H - 21.5 * mm,
        "T : +250 788 90 30 30, +250 788 90 30 32,   W : www.ines.ac.rw,   E : info@ines.ac.rw",
    )
    canvas.setFont("Times-Roman", 10)
    canvas.drawRightString(W - 18 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def build_doc(path: Path, story: list, title: str) -> None:
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=36 * mm,
        bottomMargin=18 * mm,
        title=title,
        author="Group AI-G14",
    )
    doc.build(story, onFirstPage=draw_header_footer, onLaterPages=draw_header_footer)


def uiux_story() -> list:
    usable = 178 * mm
    story = [
        P("SWE 3513", "course"),
        P("Artificial Intelligence", "course"),
        P("AI_A1_G14 UI and UX design", "title"),
        P("Musanze Cooperative Harvest and Dispatch Decision Dashboard", "title"),
        P("A proposed staff interface for HarvestLink operations  ·  Group AI-G14", "subtitle"),
        navy_table(
            [
                ["Document detail", "Content"],
                ["Group code", "AI-G14"],
                ["Group leader", "Teta Erssie"],
                [
                    "Members",
                    "Joel Eliezer Youto Mongar Jr (Member 1, Data and UX); "
                    "Niyibigira Gad (Member 2, Regression); "
                    "Uwiringiyimana Marie Claire (Member 3, Classification); "
                    "Mbabazi Sandrine (Member 4, Clustering and QA); "
                    "Teta Erssie (Member 5, Reproducibility and release).",
                ],
                [
                    "Stakeholder",
                    "HarvestLink operations team: collection clerks, dispatchers, and the floor supervisor who authorises trucks.",
                ],
                ["Document type", "Five-to-seven page staff dashboard proposal. The interface is not implemented in code."],
                ["Related commands", "python run_all.py --data data/AI_A1_G14.csv --output artifacts/ --group AI-G14<br/>python predict.py --record '{...}'"],
            ],
            [42 * mm, usable - 42 * mm],
        ),
        P("Cover and operational problem", "heading"),
        P(
            "Musanze HarvestLink Cooperative collects potatoes from several fictional farms. "
            "Before a truck leaves, non-technical staff must act on three model outputs from the same record. "
            "This PDF shows how those outputs appear on a proposed dashboard. It does not replace the command-line pipeline.",
            "body",
        ),
        navy_table(
            [
                ["Decision", "Model output the staff see", "Action the staff take"],
                [
                    "1. Estimate expected harvest weight",
                    "Predicted actual_yield_kg from NumPy linear regression, shown in kilograms with last test MAE/RMSE.",
                    "Plan crates and truck space. If confidence is low, weigh the load before dispatch.",
                ],
                [
                    "2. Flag consignments needing dispatch attention",
                    "Predicted dispatch_attention (0 or 1) and the probability of class 1 from logistic regression.",
                    "Hold class-1 loads for a supervisor check. A missed flag is more costly than an extra check.",
                ],
                [
                    "3. Group collection points with similar profiles",
                    "Cluster label from k-means on input features only, with a caution that clusters are not farm types.",
                    "Compare similar collection points. Do not treat a colour or number as a verified category.",
                ],
            ],
            [42 * mm, 68 * mm, usable - 110 * mm],
        ),
        PageBreak(),
        P("User journey and information flow", "heading"),
        P(
            "The journey starts with the lecturer-issued CSV and ends with a human decision. "
            "No truck is released by the model. The dashboard is a reading layer over run_all.py and predict.py.",
            "body",
        ),
        navy_table(
            [
                ["Step", "User action", "System response", "Human decision"],
                [
                    "1. CSV upload",
                    "Clerk selects data/AI_A1_G14.csv (schema unchanged).",
                    "Pipeline checks columns and types, reports missing values and duplicates, prints group code and SHA-256.",
                    "Accept the file only if the fingerprint and schema match. Reject renamed or mixed files.",
                ],
                [
                    "2. Quality review",
                    "Clerk opens the data-quality screen.",
                    "Shows row count, feature count, missingness, and that record_id is not a model feature.",
                    "Stop if the schema fails. Continue only when the quality gate is green.",
                ],
                [
                    "3. Training run",
                    "Analyst runs run_all.py from the README command.",
                    "Writes metrics, plots, and models into artifacts/ and models/. Overwrites previous outputs.",
                    "Read MAE, confusion matrix, and selected k. Do not type metrics by hand.",
                ],
                [
                    "4. Prediction",
                    "Dispatcher enters one farm record (six input fields).",
                    "predict.py returns JSON: yield kg, class, probability, cluster, group code, model version.",
                    "Read the numbers on the prediction screen. Incomplete JSON is blocked, not filled in.",
                ],
                [
                    "5. Human decision",
                    "Supervisor chooses Confirm, Hold, or Override.",
                    "Stores the human choice with the model advice attached.",
                    "The supervisor’s decision is authoritative. Class 1 never auto-dispatches.",
                ],
            ],
            [22 * mm, 40 * mm, 58 * mm, usable - 120 * mm],
        ),
        P("Information flow (summary)", "heading"),
        P(
            "CSV file → schema and SHA-256 gate → feature matrix (identifiers and targets removed) → "
            "regression / classification / clustering artefacts → single-record JSON → staff review → "
            "Confirm / Hold / Override. Targets actual_yield_kg and dispatch_attention are never clustering inputs. "
            "record_id is never a model feature.",
            "body",
        ),
        PageBreak(),
        P("Annotated wireframes", "heading"),
        P(
            "Pages 3 and 4 show the four required screens: data quality, prediction, classification warning, and clusters. "
            "Callouts in the tables under each figure state what a non-technical user should do.",
            "body",
        ),
        P("Figure 1. Data quality review", "heading"),
        screen_frame(
            "HarvestLink  |  Data quality  |  Group AI-G14",
            [
                [P("<b>Dataset fingerprint (SHA-256)</b><br/>Shown in full and copied to Moodle. Must match data_report.json.", "screen_body")],
                [P("<b>Schema checklist</b><br/>record_id, plot_area_ha, rainfall_mm, soil_ph, seed_kg, distance_km, arrival_hour, actual_yield_kg, dispatch_attention.", "screen_body")],
                [P("<b>Counts (computed, not typed)</b><br/>Row count  ·  Feature count  ·  Missing values  ·  Duplicate rows  ·  Duplicate record_id", "screen_body")],
                [P("<b>Gate</b><br/>[ Reject file — schema mismatch ]     [ Continue to models ]<br/>If columns differ, training is blocked.", "screen_body")],
            ],
            [usable],
        ),
        navy_table(
            [
                ["Callout", "What the user understands"],
                ["A. Fingerprint", "This is the same file the lecturer issued. A different hash means a replaced or edited CSV."],
                ["B. Schema list", "Only the published columns are allowed. Extra or renamed fields stop the run."],
                ["C. Counts", "Quality numbers come from the pipeline. Staff do not paste metrics into the screen."],
                ["D. Gate", "Prediction is unavailable until the file passes. This prevents acting on a broken upload."],
            ],
            [36 * mm, usable - 36 * mm],
        ),
        Spacer(1, 8),
        P("Figure 2. Harvest prediction", "heading"),
        screen_frame(
            "HarvestLink  |  Score one consignment  |  predict.py",
            [
                [P("<b>Input fields</b>  plot_area_ha    rainfall_mm    soil_ph    seed_kg    distance_km    arrival_hour", "screen_body")],
                [P("<b>Command</b>  python predict.py --record '{...six fields...}'", "screen_body")],
                [P("<b>Returned JSON (read-only)</b><br/>regression_prediction_kg<br/>classification_prediction + classification_probability<br/>cluster_label  ·  group_code  ·  model_version", "screen_body")],
                [P("<b>Staff reading</b><br/>Predicted yield: [ kg ]    Last test MAE / RMSE shown beside the number.<br/>Low R² or a very small test set: display “low confidence — check the scale”.", "screen_body")],
            ],
            [usable],
        ),
        navy_table(
            [
                ["Callout", "What the user understands"],
                ["E. Six inputs only", "The dispatcher types farm measurements. record_id is not required for scoring."],
                ["F. JSON contract", "The screen shows the same keys the assessor will see in the terminal."],
                ["G. kg plus error", "A point estimate without MAE/RMSE would look more precise than the model is."],
            ],
            [36 * mm, usable - 36 * mm],
        ),
        PageBreak(),
        P("Figure 3. Classification warning", "heading"),
        screen_frame(
            "HarvestLink  |  Dispatch attention",
            [
                [P("<b>WARNING — ATTENTION REQUIRED</b><br/>Predicted class: 1     Probability of attention: shown as 0–1, not as a percentage slogan.", "screen_body")],
                [P("<b>Cost rule shown on screen</b><br/>False negative (missed class 1) is more costly than false positive (extra check).<br/>Class 1 = hold for supervisor. Class 0 = proceed with normal checks.", "screen_body")],
                [P("<b>Last training confusion matrix</b><br/>TN | FP<br/>FN (costly) | TP", "screen_body")],
                [P("<b>Actions</b>  [ Hold truck ]   [ Confirm class 0 ]   [ Override — reason required ]", "screen_body")],
            ],
            [usable],
        ),
        navy_table(
            [
                ["Callout", "What the user understands"],
                ["H. Class and probability together", "0.51 is not the same as 0.95. Borderline scores invite Hold, not autopilot."],
                ["I. FN highlighted", "Missing a load that needed attention can send an at-risk consignment. Extra review is cheaper."],
                ["J. Override", "The model advises. A named supervisor records why the advice was not followed."],
            ],
            [42 * mm, usable - 42 * mm],
        ),
        Spacer(1, 8),
        P("Figure 4. Collection-point clusters", "heading"),
        screen_frame(
            "HarvestLink  |  Operating profiles  |  unsupervised k-means",
            [
                [P("<b>Plot</b>  PCA view of input features only. Colour = selected k from silhouette (k compared from 2 to 5).", "screen_body")],
                [P("<b>Legend</b>  Cluster 0, Cluster 1, …   Label: “similar input profile”, not “good farm” or “bad farm”.", "screen_body")],
                [P("<b>On-screen caution</b>  Clusters are groupings in this file. They are not verified real-world categories.", "screen_body")],
                [P("<b>New record</b>  Assigned to the nearest centroid. No target columns are used.", "screen_body")],
            ],
            [usable],
        ),
        navy_table(
            [
                ["Callout", "What the user understands"],
                ["K. Similarity only", "Nearby points share plot, rain, pH, seed, distance, and arrival patterns."],
                ["L. No ranking", "A cluster number is not a performance grade and must not be used as one."],
                ["M. Targets excluded", "actual_yield_kg and dispatch_attention are not clustering inputs."],
            ],
            [36 * mm, usable - 36 * mm],
        ),
        PageBreak(),
        P("Responsible AI states", "heading"),
        P(
            "The dashboard must remain usable when the model is uncertain, when data are missing, when the method is limited, and when a person disagrees with the score. "
            "These four states are required evidence for this page.",
            "body",
        ),
        navy_table(
            [
                ["State", "What the staff see", "What the system does"],
                [
                    "Uncertainty",
                    "Predicted kg with MAE/RMSE. Attention shown as class plus probability. Banner “low confidence” when test R² is weak or the test split is tiny.",
                    "Does not hide intervals or probabilities. Does not present a single kilogram as a certified weight.",
                ],
                [
                    "Missing or malformed data",
                    "Red list of missing field names, or “Malformed JSON”. No invented defaults for soil_ph or other inputs.",
                    "predict.py rejects the record and returns a JSON error. Training stops if the CSV schema does not match the published columns.",
                ],
                [
                    "Model limitation",
                    "Plain-language note: linear yield, logistic attention, k-means profiles. “This is advice, not a farm verdict.”",
                    "record_id is never a feature. Supervised scalers fit on training rows only. Hidden assessor CSVs must recompute metrics.",
                ],
                [
                    "Human override",
                    "Buttons Confirm, Hold, Override. Override requires a short reason (weather, late truck, damaged crate).",
                    "Stores the human decision as the official outcome. The model output remains attached as advice. No automatic dispatch from class 1.",
                ],
            ],
            [32 * mm, 73 * mm, usable - 105 * mm],
        ),
        P(
            "Live verification note. Named constants live in src/config.py (RANDOM_SEED, LEARNING_RATE, N_ITERATIONS, K_MIN, K_MAX). "
            "A member can change one value and explain the new artefacts without editing hidden row counts.",
            "body",
        ),
        PageBreak(),
        P("Visual system and rationale", "heading"),
        P(
            "Colour, labels, and legends follow the same navy-and-table language as this assignment document so staff are not asked to learn a separate brand. "
            "Colour is never the only signal.",
            "body",
        ),
        navy_table(
            [
                ["Element", "Specification", "Accessibility reason"],
                [
                    "Primary navy #17365D",
                    "Screen titles, table headers, primary buttons.",
                    "High contrast on white. Matches the official INES table header in this assignment format.",
                ],
                [
                    "Alert amber #C43E00 on #FFF4E5",
                    "Dispatch-attention banner and false-negative cell.",
                    "Not red-only: the banner also contains the word REQUIRED and the class number.",
                ],
                [
                    "Body text black on white",
                    "All reading text, including kg values.",
                    "Contrast above typical 7:1. No light-grey body copy.",
                ],
                [
                    "Labels",
                    "“Predicted kg”, “Attention probability”, “Cluster (unsupervised)”.",
                    "Verbs on buttons: Upload, Score, Hold, Override. No “safe farm” label.",
                ],
                [
                    "Legends",
                    "Cluster colours named Cluster 0, Cluster 1, with the caution sentence under the plot.",
                    "Colour-blind users still have the numeric label and the caution text.",
                ],
            ],
            [42 * mm, 62 * mm, usable - 104 * mm],
        ),
        P("Three justified product decisions", "heading"),
        navy_table(
            [
                ["Decision", "Justification"],
                [
                    "1. Show kilograms together with MAE/RMSE",
                    "A lone predicted kilogram invites over-trust. Staff plan crates from a number plus an error scale, then weigh when confidence is low.",
                ],
                [
                    "2. Pair class 1 with probability and a Hold action",
                    "A binary flag hides borderline cases. Probability plus Hold/Override keeps a person in the loop when the score is near 0.5. False negatives remain the costly error.",
                ],
                [
                    "3. Name clusters as profiles, never as grades",
                    "Silhouette only ranks compactness of groups in feature space. Calling a cluster poor or rich would be a claim the method cannot support, and the assignment forbids treating clusters as verified real-world categories.",
                ],
            ],
            [52 * mm, usable - 52 * mm],
        ),
        P("Page checklist against the assignment table", "heading"),
        navy_table(
            [
                ["Page", "Required content", "Where it is in this PDF"],
                ["1", "Cover and operational problem", "Group code, members, stakeholder, three decisions."],
                ["2", "User journey and information flow", "CSV upload → review → prediction → human decision."],
                ["3 to 4", "Annotated wireframes", "Data quality, prediction, classification warning, clusters."],
                ["5", "Responsible AI states", "Uncertainty, missing data, model limitation, human override."],
                ["6", "Visual system and rationale", "Colours, labels, legends, three justified decisions."],
            ],
            [28 * mm, 52 * mm, usable - 80 * mm],
        ),
    ]
    return story


def contributions_story() -> list:
    usable = 178 * mm
    return [
        P("SWE 3513", "course"),
        P("Artificial Intelligence", "course"),
        P("AI_A1_G14 contribution and accountability record", "title"),
        P("Signed mapping of requirements to members  ·  Group AI-G14", "subtitle"),
        navy_table(
            [
                ["Document detail", "Content"],
                ["Group code", "AI-G14"],
                ["Group leader", "Teta Erssie"],
                ["Dataset", "data/AI_A1_G14.csv (lecturer-issued; not renamed or replaced)"],
                ["Related files", "AI_A1_G14.zip, AI_A1_G14_UIUX.pdf, evidence/AI_USE.md"],
            ],
            [42 * mm, usable - 42 * mm],
        ),
        P("Role mapping", "heading"),
        navy_table(
            [
                ["Role", "Student", "Primary files", "Requirement owned"],
                [
                    "Member 5<br/>Reproducibility",
                    "Teta Erssie",
                    "run_all.py, predict.py, README.md, requirements.txt, evidence packaging",
                    "CLI contract, clean run, final hash, predict validation",
                ],
                [
                    "Member 1<br/>Data and UX",
                    "Joel Eliezer Youto Mongar Jr",
                    "src/data.py, AI_A1_G14_UIUX.pdf",
                    "Schema, vectorization, data report, dashboard PDF",
                ],
                [
                    "Member 2<br/>Regression",
                    "Niyibigira Gad",
                    "src/regression.py, evidence/REGRESSION_DERIVATION.md",
                    "NumPy gradient descent, scaling, loss, MAE/RMSE/R²",
                ],
                [
                    "Member 3<br/>Classification",
                    "Uwiringiyimana Marie Claire",
                    "src/classification.py, confusion_matrix.png",
                    "Split, metrics, confusion matrix, cost of false negatives",
                ],
                [
                    "Member 4<br/>Clustering and QA",
                    "Mbabazi Sandrine",
                    "src/clustering.py, evidence/TEST_LOG.pdf",
                    "k = 2 to 5, silhouette, labels, test log",
                ],
            ],
            [32 * mm, 38 * mm, 52 * mm, usable - 122 * mm],
        ),
        P(
            "Each member must make at least two meaningful commits on the owned files and must be able to change seed, learning rate, threshold, or k range in the live check. Cosmetic commits do not count.",
            "body",
        ),
        PageBreak(),
        P("Integrity statement", "heading"),
        P(
            "We confirm that the lecturer-issued dataset was not renamed, extended, or replaced; that files inside artifacts/ were generated by the submitted code; and that generative AI use is disclosed in evidence/AI_USE.md. "
            "A member who cannot explain the listed files receives zero for individual verification even if the group ZIP runs.",
            "body",
        ),
        navy_table(
            [
                ["Student", "Registration number", "Signature", "Date", "Commit hash (owned work)"],
                ["Teta Erssie", "", "", "", ""],
                ["Joel Eliezer Youto Mongar Jr", "", "", "", ""],
                ["Niyibigira Gad", "", "", "", ""],
                ["Uwiringiyimana Marie Claire", "", "", "", ""],
                ["Mbabazi Sandrine", "", "", "", ""],
            ],
            [42 * mm, 32 * mm, 38 * mm, 28 * mm, usable - 140 * mm],
        ),
        Spacer(1, 10),
        P("Peer accountability", "heading"),
        P(
            "Confidential peer ratings must be consistent with this mapping and with the Git history. "
            "Only the group leader submits on Moodle. The GitHub commit hash entered in Moodle must match the ZIP.",
            "body",
        ),
    ]


def main() -> None:
    uiux = ROOT / "AI_A1_G14_UIUX.pdf"
    contrib = ROOT / "AI_A1_G14_CONTRIBUTIONS.pdf"
    build_doc(uiux, uiux_story(), "AI_A1_G14 UI and UX design")
    build_doc(contrib, contributions_story(), "AI_A1_G14 contributions")
    print(f"Wrote {uiux}")
    print(f"Wrote {contrib}")


if __name__ == "__main__":
    main()
