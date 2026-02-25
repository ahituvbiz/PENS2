import streamlit as st
import fitz
import json
import os
import math
import pandas as pd
import re
from openai import OpenAI

# הגדרות RTL ועיצוב קשיח - חסימת כל אפשרות לעיגול או פרשנות
st.set_page_config(page_title="מנתח פנסיה - גירסה 30.0 (דיוק מוחלט)", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@400;700&display=swap');
    * { font-family: 'Assistant', sans-serif; direction: rtl; text-align: right; }
    .stTable { direction: rtl !important; width: 100%; }
    th, td { text-align: right !important; padding: 12px !important; white-space: nowrap; }
    .val-success { padding: 12px; border-radius: 8px; margin-bottom: 10px; font-weight: bold; background-color: #f0fdf4; border: 1px solid #16a34a; color: #16a34a; }
    .val-error { padding: 12px; border-radius: 8px; margin-bottom: 10px; font-weight: bold; background-color: #fef2f2; border: 1px solid #dc2626; color: #dc2626; }
    .info-box { padding: 14px; border-radius: 8px; margin-bottom: 10px; font-weight: bold; background-color: #eff6ff; border: 1px solid #2563eb; color: #1d4ed8; }
    .warn-box { padding: 14px; border-radius: 8px; margin-bottom: 10px; font-weight: bold; background-color: #fffbeb; border: 1px solid #d97706; color: #92400e; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# קבועים גלובליים
# ============================================================

# ✅ שיפור 1: סכמת JSON קבועה ומוגדרת במקום אחד
# שינוי זה מונע חוסר עקביות בין הפרומפט לבין מה שהקוד מצפה לקבל
JSON_SCHEMA = {
    "table_a": {"rows": [{"תיאור": "", "סכום בש\"ח": ""}]},
    "table_b": {"rows": [{"תיאור": "", "סכום בש\"ח": ""}]},
    "table_c": {"rows": [{"תיאור": "", "אחוז": ""}]},
    "table_d": {"rows": [{"מסלול": "", "תשואה": ""}]},
    "table_e": {"rows": [{"שם המעסיק": "", "מועד": "", "חודש": "", "שכר": "", "עובד": "", "מעסיק": "", "פיצויים": "", "סה\"כ": ""}]}
}

# ✅ שיפור 2: הגדרת הפרומפט כקבוע נפרד – שינוי בפרומפט לא ישבור את שאר הקוד
# הפרומפט מחוזק עם דוגמאות מפורשות של מה שאסור לעשות
EXTRACTION_SYSTEM_PROMPT = """You are a MECHANICAL CHARACTER COPIER. 
Rules that CANNOT be broken:
1. Copy digits exactly as they appear. If you see 67, output 67. NEVER output 76.
2. NEVER round. 0.17 stays 0.17. NEVER output 1.0 or 0.2.
3. NEVER infer or guess missing values. If a cell is empty, output "".
4. NEVER merge rows or split rows.
5. Output ONLY valid JSON. No markdown, no explanation, no preamble."""

EXTRACTION_USER_PROMPT_TEMPLATE = """Copy the following pension report tables into the exact JSON schema below.

FORBIDDEN ACTIONS (will cause system failure):
- Rounding any number (0.17 must remain 0.17, not 0.2)
- Swapping digits (67 must remain 67, not 76)
- Adding rows that don't exist in the text
- Removing rows that exist in the text
- Filling empty cells with guesses

REQUIRED JSON SCHEMA:
{schema}

PENSION REPORT TEXT:
{text}"""

MAX_RETRIES = 3  # ✅ שיפור 3: מספר ניסיונות חוזרים אם ולידציה נכשלת

# ============================================================
# אתחול לקוח
# ============================================================

def init_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    return OpenAI(api_key=api_key) if api_key else None

def clean_num(val):
    if val is None or val == "" or str(val).strip() in ["-", "nan", ".", "0"]: return 0.0
    try:
        cleaned = re.sub(r'[^\d\.\-]', '', str(val).replace(",", "").replace("−", "-"))
        return float(cleaned) if cleaned else 0.0
    except: return 0.0

# ============================================================
# סינון דוחות לא רלוונטיים
# ============================================================

def is_vector_pdf(pdf_doc):
    total_chars = sum(len(page.get_text().strip()) for page in pdf_doc)
    return total_chars > 100

def check_clal(text):
    clal_keywords = ["כלל ביטוח", "כלל פנסיה", "כלל חברה", "Clal"]
    return any(kw in text for kw in clal_keywords)

def extract_title_lines(pdf_doc, max_lines=10):
    first_page_text = pdf_doc[0].get_text() if len(pdf_doc) > 0 else ""
    non_empty_lines = [l.strip() for l in first_page_text.splitlines() if l.strip()]
    return non_empty_lines[:max_lines]

def check_comprehensive_pension(pdf_doc, table_a_rows):
    title_lines = extract_title_lines(pdf_doc)
    title_text = " ".join(title_lines)
    if "כללית" in title_text or "יסוד" in title_text:
        return False, "כותרת הדוח מכילה 'כללית' או 'יסוד'"
    if len(table_a_rows) < 6:
        return False, f"טבלא א' מכילה {len(table_a_rows)} שורות בלבד (נדרשות לפחות 6)"
    return True, ""

def run_filters(pdf_doc, raw_text, table_a_rows, employment_type):
    if len(pdf_doc) > 4:
        return False, "הרובוט בוחן רק דוחות מקוצרים של קרן פנסיה מקיפה."
    is_comprehensive, reason = check_comprehensive_pension(pdf_doc, table_a_rows)
    if not is_comprehensive:
        return False, f"הרובוט בוחן רק דוחות מקוצרים של קרן פנסיה מקיפה. ({reason})"
    if not is_vector_pdf(pdf_doc):
        return False, "נא העלה קובץ מקורי אותו הורדת מאתר החברה."
    if employment_type != "שכיר":
        return False, "עדיין לא למדתי לנתח דוחות של מי שאיננו שכיר בלבד."
    if check_clal(raw_text):
        return False, "יש לי קושי לקרוא את הדוחות של חברת כלל. נסה שוב בקרוב."
    return True, ""

# ============================================================
# ✅ שיפור 4: ולידציה מורחבת – כל טבלה נבדקת בנפרד
# ============================================================

def validate_extracted_data(data):
    """
    מחזיר (is_valid: bool, errors: list[str])
    בודק:
    - כל טבלה קיימת ולא ריקה
    - טבלא א': לפחות שורה אחת עם מספר חיובי
    - טבלא ב': לפחות שורה אחת עם מספר חיובי
    - טבלא ה': שורת סיכום עם סה"כ > 0 ושכר > 0
    """
    errors = []

    for table_key in ["table_a", "table_b", "table_c", "table_d", "table_e"]:
        rows = data.get(table_key, {}).get("rows", [])
        if not rows:
            errors.append(f"טבלה {table_key} ריקה")

    # טבלא א': לפחות ערך כספי אחד חיובי
    rows_a = data.get("table_a", {}).get("rows", [])
    if not any(clean_num(r.get("סכום בש\"ח", 0)) > 0 for r in rows_a):
        errors.append("טבלא א': אין ערכים כספיים חיוביים")

    # טבלא ב': לפחות ערך כספי אחד חיובי
    rows_b = data.get("table_b", {}).get("rows", [])
    if not any(clean_num(r.get("סכום בש\"ח", 0)) > 0 for r in rows_b):
        errors.append("טבלא ב': אין ערכים כספיים חיוביים")

    # טבלא ה': שורת סיכום תקינה
    rows_e = data.get("table_e", {}).get("rows", [])
    if rows_e:
        last = rows_e[-1]
        total = clean_num(last.get("סה\"כ", 0))
        salary = clean_num(last.get("שכר", 0))
        if total <= 0:
            errors.append("טבלא ה': שורת סיכום – סה\"כ = 0")
        if salary <= 0:
            errors.append("טבלא ה': שורת סיכום – שכר = 0")

    return len(errors) == 0, errors


def perform_cross_validation(data):
    """אימות הצלבה קשיח בין טבלה ב' ל-ה'"""
    dep_b = 0.0
    for r in data.get("table_b", {}).get("rows", []):
        row_str = " ".join(str(v) for v in r.values())
        if any(kw in row_str for kw in ["הופקדו", "כספים שהופקדו"]):
            nums = [clean_num(v) for v in r.values() if clean_num(v) > 10]
            if nums: dep_b = nums[0]
            break

    rows_e = data.get("table_e", {}).get("rows", [])
    dep_e = clean_num(rows_e[-1].get("סה\"כ", 0)) if rows_e else 0.0

    if abs(dep_b - dep_e) < 5 and dep_e > 0:
        st.markdown(f'<div class="val-success">✅ אימות הצלבה עבר: סכום ההפקדות ({dep_e:,.2f} ₪) תואם במדויק.</div>', unsafe_allow_html=True)
    elif dep_e > 0:
        st.markdown(f'<div class="val-error">⚠️ שגיאת אימות: טבלה ב\' ({dep_b:,.2f} ₪) לעומת טבלה ה\' ({dep_e:,.2f} ₪).</div>', unsafe_allow_html=True)

def display_pension_table(rows, title, col_order):
    if not rows: return
    df = pd.DataFrame(rows)
    existing = [c for c in col_order if c in df.columns]
    df = df[existing]
    df.index = range(1, len(df) + 1)
    st.subheader(title)
    st.table(df)

# ============================================================
# ✅ שיפור המרכזי: חילוץ עם ניסיונות חוזרים + seed קבוע
# ============================================================

def call_openai_extraction(client, text, attempt=0):
    """
    קריאה בודדת ל-API עם:
    - temperature=0: מבטל אקראיות
    - seed=42: מבטיח שאותו קלט → אותו פלט (תכונה של OpenAI)
    - response_format=json_object: מונע טקסט מיותר סביב ה-JSON
    """
    prompt = EXTRACTION_USER_PROMPT_TEMPLATE.format(
        schema=json.dumps(JSON_SCHEMA, ensure_ascii=False, indent=2),
        text=text
    )
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0,      # ✅ ביטול אקראיות
        seed=42,            # ✅ חדש: גורם לאותו קלט → אותו פלט בכל הרצה
        response_format={"type": "json_object"}
    )
    return json.loads(res.choices[0].message.content)


def process_audit_v30(client, text):
    """
    ✅ שיפור 3: לוגיקת ניסיונות חוזרים (retry)
    אם הולידציה נכשלת – ננסה שוב עד MAX_RETRIES פעמים.
    כך אנחנו מגנים מפני כשלונות חד-פעמיים של המודל.
    """
    data = None
    last_errors = []

    for attempt in range(MAX_RETRIES):
        try:
            data = call_openai_extraction(client, text, attempt)
        except Exception as e:
            last_errors = [f"שגיאת API: {e}"]
            continue

        is_valid, errors = validate_extracted_data(data)

        if is_valid:
            if attempt > 0:
                st.markdown(f'<div class="val-success">✅ חילוץ הצליח בניסיון מספר {attempt + 1}.</div>', unsafe_allow_html=True)
            break
        else:
            last_errors = errors
            if attempt < MAX_RETRIES - 1:
                st.markdown(f'<div class="warn-box">⚠️ ניסיון {attempt + 1} נכשל ({", ".join(errors)}). מנסה שוב...</div>', unsafe_allow_html=True)

    if data is None or last_errors:
        st.markdown(f'<div class="val-error">❌ החילוץ נכשל לאחר {MAX_RETRIES} ניסיונות: {", ".join(last_errors)}</div>', unsafe_allow_html=True)
        return None

    # ── תיקון הסטות וחישוב שכר ב-Python (ללא AI) ──
    rows_e = data.get("table_e", {}).get("rows", [])
    if len(rows_e) > 1:
        last_row = rows_e[-1]

        salary_sum = sum(clean_num(r.get("שכר", 0)) for r in rows_e[:-1])

        vals = [last_row.get("עובד"), last_row.get("מעסיק"), last_row.get("פיצויים"), last_row.get("סה\"כ")]
        cleaned_vals = [clean_num(v) for v in vals]
        max_val = max(cleaned_vals)

        if max_val > 0 and clean_num(last_row.get("סה\"כ")) != max_val:
            non_zero_vals = [v for v in vals if clean_num(v) > 0]
            if len(non_zero_vals) == 4:
                last_row["סה\"כ"] = non_zero_vals[3]
                last_row["פיצויים"] = non_zero_vals[2]
                last_row["מעסיק"] = non_zero_vals[1]
                last_row["עובד"] = non_zero_vals[0]
            elif len(non_zero_vals) == 3:
                last_row["סה\"כ"] = non_zero_vals[2]
                last_row["מעסיק"] = non_zero_vals[1]
                last_row["עובד"] = non_zero_vals[0]
                last_row["פיצויים"] = "0"

        last_row["שכר"] = f"{salary_sum:,.0f}"
        last_row["מועד"] = ""
        last_row["חודש"] = ""
        last_row["שם המעסיק"] = "סה\"כ"

    return data

# ============================================================
# חישוב שנים לפרישה והכנסה מבוטחת
# ============================================================

def calc_nper(rate_annual, pv, fv):
    if pv <= 0 or fv <= 0:
        return None
    try:
        n = math.log(fv / pv) / math.log(1 + rate_annual)
        return round(n, 1)
    except (ValueError, ZeroDivisionError):
        return None

def calc_years_to_retirement_and_insured_income(data):
    st.subheader("📊 ניתוח פיננסי")

    rows_a = data.get("table_a", {}).get("rows", [])
    rows_b = data.get("table_b", {}).get("rows", [])

    monthly_pension = clean_num(rows_a[0].get("סכום בש\"ח", 0)) if rows_a else 0.0
    fv_target = monthly_pension * 190
    current_balance = clean_num(rows_b[-1].get("סכום בש\"ח", 0)) if rows_b else 0.0

    years = calc_nper(0.0386, current_balance, fv_target)

    if years is not None:
        st.markdown(f'<div class="info-box">📅 שנים לפרישה (לפי תשואה שנתית של 3.86%): <b>{years}</b> שנים</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="val-error">⚠️ לא ניתן לחשב שנים לפרישה – חסרים נתונים מטבלאות א\' ו-ב\'.</div>', unsafe_allow_html=True)

    rows_e = data.get("table_e", {}).get("rows", [])
    if not rows_e:
        st.markdown('<div class="val-error">⚠️ אין נתונים בטבלא ה\' לחישוב הכנסה מבוטחת.</div>', unsafe_allow_html=True)
        return

    last_e = rows_e[-1]
    total_deposits = clean_num(last_e.get("סה\"כ", 0))
    total_salary = clean_num(last_e.get("שכר", 0))

    if total_salary == 0:
        st.markdown('<div class="val-error">⚠️ לא ניתן לחשב שיעור הפקדה – סה"כ שכר הוא 0.</div>', unsafe_allow_html=True)
        return

    deposit_rate = total_deposits / total_salary

    if not (0.185 <= deposit_rate <= 0.2283):
        st.markdown(f'<div class="val-error">⚠️ שיעור הפקדה: {deposit_rate*100:.2f}% – חורג מהטווח הצפוי (18.5%–22.83%). בדוק את הנתונים.</div>', unsafe_allow_html=True)

    waiver_value = clean_num(rows_a[-1].get("סכום בש\"ח", 0)) if rows_a else 0.0
    insured_deposit = waiver_value / 0.94 if waiver_value > 0 else 0.0
    insured_income = insured_deposit / deposit_rate if deposit_rate > 0 else 0.0

    st.markdown(f'<div class="info-box">💼 הכנסה מבוטחת לפי שחרור: <b>{insured_income:,.2f} ₪</b></div>', unsafe_allow_html=True)

    SURVIVOR_SPOUSE_KEYWORDS = ["אלמן", "אלמנה", "שאר", "בן זוג"]
    SURVIVOR_ORPHAN_KEYWORDS  = ["יתום", "ילד"]

    def find_row_by_keywords(rows, keywords):
        for row in rows:
            desc = str(row.get("תיאור", ""))
            if any(kw in desc for kw in keywords):
                return clean_num(row.get("סכום בש\"ח", 0))
        return None

    spouse_pension  = find_row_by_keywords(rows_a, SURVIVOR_SPOUSE_KEYWORDS)
    orphan_pension  = find_row_by_keywords(rows_a, SURVIVOR_ORPHAN_KEYWORDS)

    survivors_total = None
    if spouse_pension is not None and orphan_pension is not None:
        survivors_total = spouse_pension + orphan_pension
        st.markdown(f'<div class="info-box">👨‍👩‍👧 הכנסה מבוטחת לפי שארים: <b>{survivors_total:,.2f} ₪</b></div>', unsafe_allow_html=True)
    else:
        missing = []
        if spouse_pension is None: missing.append("קצבת אלמן/ה")
        if orphan_pension is None: missing.append("קצבת יתום")
        st.markdown(f'<div class="warn-box">⚠️ לא נמצאו בטבלא א\' הערכים הבאים: {", ".join(missing)}. לא ניתן לחשב הכנסה מבוטחת לפי שארים.</div>', unsafe_allow_html=True)

    DISABILITY_KEYWORDS = ["נכות", "אובדן כושר", "כושר עבודה"]
    disability_pension = find_row_by_keywords(rows_a, DISABILITY_KEYWORDS)

    insured_income_disability = None
    if disability_pension is not None:
        insured_income_disability = disability_pension / 0.75
        st.markdown(f'<div class="info-box">🏥 הכנסה מבוטחת לפי נכות: <b>{insured_income_disability:,.2f} ₪</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">⚠️ לא נמצאה שורת קצבת נכות בטבלא א\'. לא ניתן לחשב הכנסה מבוטחת לפי נכות.</div>', unsafe_allow_html=True)

    if survivors_total is not None and insured_income_disability is not None:
        if abs(survivors_total - insured_income_disability) > 1:
            st.markdown(
                f'<div class="val-error">⚠️ שים לב: הכנסה מבוטחת לפי שארים ({survivors_total:,.2f} ₪) '
                f'שונה מהכנסה מבוטחת לפי נכות ({insured_income_disability:,.2f} ₪).</div>',
                unsafe_allow_html=True
            )

    if insured_income > 0 and insured_income_disability is not None and insured_income_disability > 0:
        diff_pct = abs(insured_income - insured_income_disability) / insured_income
        if diff_pct > 0.10:
            st.markdown(
                f'<div class="val-error">⚠️ שים לב: קיים הפרש של {diff_pct*100:.1f}% בין הכנסה מבוטחת לפי שחרור '
                f'({insured_income:,.2f} ₪) לבין הכנסה מבוטחת לפי נכות ({insured_income_disability:,.2f} ₪).</div>',
                unsafe_allow_html=True
            )

# ============================================================
# ממשק משתמש
# ============================================================
st.title("📋 חילוץ נתונים פנסיוני - גירסה 30.0")
client = init_client()

if client:

    st.subheader("פרטי הלקוח")
    col1, col2, col3 = st.columns(3)

    with col1:
        employment_type = st.radio("סטטוס תעסוקתי", options=["שכיר", "עצמאי", "שכיר + עצמאי"], index=0, horizontal=False)

    with col2:
        gender = st.radio("מגדר", options=["גבר", "אשה"], index=0, horizontal=False)

    with col3:
        marital_status = st.radio("מצב משפחתי", options=["נשוי/אה", "רווק/ה", "גרוש/ה", "אלמן/ה"], index=0, horizontal=False)

    has_young_children = None
    if marital_status in ["גרוש/ה", "אלמן/ה"]:
        has_young_children = st.radio("האם יש לך ילדים מתחת לגיל 21?", options=["כן", "לא"], index=0, horizontal=True)

    st.markdown("---")

    file = st.file_uploader("העלה דוח PDF", type="pdf")
    if file:
        with st.spinner("מעתיק נתונים כפי שהם (ללא שיקול דעת AI)..."):
            file_bytes = file.read()
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            raw_text = "\n".join([page.get_text() for page in pdf_doc])

            temp_table_a_rows = [
                line for line in raw_text.splitlines()
                if line.strip() and any(c.isdigit() for c in line)
            ]

            passed, filter_msg = run_filters(pdf_doc, raw_text, temp_table_a_rows, employment_type)

            if not passed:
                st.error(filter_msg)
            else:
                data = process_audit_v30(client, raw_text)

                if data:
                    perform_cross_validation(data)
                    calc_years_to_retirement_and_insured_income(data)
                    st.markdown("---")
                    display_pension_table(data.get("table_a", {}).get("rows"), "א. תשלומים צפויים", ["תיאור", "סכום בש\"ח"])
                    display_pension_table(data.get("table_b", {}).get("rows"), "ב. תנועות בקרן", ["תיאור", "סכום בש\"ח"])
                    display_pension_table(data.get("table_c", {}).get("rows"), "ג. דמי ניהול והוצאות", ["תיאור", "אחוז"])
                    display_pension_table(data.get("table_d", {}).get("rows"), "ד. מסלולי השקעה", ["מסלול", "תשואה"])
                    display_pension_table(data.get("table_e", {}).get("rows"), "ה. פירוט הפקדות", ["שם המעסיק", "מועד", "חודש", "שכר", "עובד", "מעסיק", "פיצויים", "סה\"כ"])
