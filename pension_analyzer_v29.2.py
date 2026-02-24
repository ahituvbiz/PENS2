import streamlit as st
import fitz
import json
import os
import math
import pandas as pd
import re
from openai import OpenAI

# הגדרות RTL ועיצוב קשיח - חסימת כל אפשרות לעיגול או פרשנות
st.set_page_config(page_title="מנתח פנסיה - גירסה 29.0 (דיוק מוחלט)", layout="wide")

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
    """בדיקה אם ה-PDF וקטורי (טקסט ניתן לחילוץ) ולא סרוק"""
    total_chars = sum(len(page.get_text().strip()) for page in pdf_doc)
    return total_chars > 100  # אם יש פחות מ-100 תווים בכל המסמך – כנראה סרוק

def check_clal(text):
    """בדיקה אם הדוח שייך לחברת כלל"""
    clal_keywords = ["כלל ביטוח", "כלל פנסיה", "כלל חברה", "Clal"]
    return any(kw in text for kw in clal_keywords)

def check_is_employee_only(text):
    """בדיקה אם הלקוח שכיר בלבד (אין רשומות של עצמאי/אובדן כושר עצמאי)"""
    self_employed_keywords = ["עצמאי", "שכר עצמאי", "הפקדת עצמאי"]
    return not any(kw in text for kw in self_employed_keywords)

def check_comprehensive_pension(text, table_a_rows):
    """
    בדיקה אם הדוח הוא של קרן פנסיה מקיפה:
    - טבלא א' חייבת לכלול לפחות 6 שורות מתחת לכותרת
    - הכותרת אסור שתכיל את המילים 'כללית' או 'יסוד'
    """
    if "כללית" in text or "יסוד" in text:
        return False, "כותרת הדוח מכילה 'כללית' או 'יסוד'"
    if len(table_a_rows) < 6:
        return False, f"טבלא א' מכילה {len(table_a_rows)} שורות בלבד (נדרשות לפחות 6)"
    return True, ""

def run_filters(pdf_doc, raw_text, table_a_rows):
    """
    מריץ את 5 מסנני הסינון לפי הסדר.
    מחזיר (passed: bool, message: str).
    """
    # מסנן 1: יותר מ-4 עמודים
    if len(pdf_doc) > 4:
        return False, "הרובוט בוחן רק דוחות מקוצרים של קרן פנסיה מקיפה."

    # מסנן 2: בדיקת קרן פנסיה מקיפה (על בסיס טקסט גולמי וטבלא א')
    is_comprehensive, reason = check_comprehensive_pension(raw_text, table_a_rows)
    if not is_comprehensive:
        return False, f"הרובוט בוחן רק דוחות מקוצרים של קרן פנסיה מקיפה. ({reason})"

    # מסנן 3: וקטורי
    if not is_vector_pdf(pdf_doc):
        return False, "נא העלה קובץ מקורי אותו הורדת מאתר החברה."

    # מסנן 4: שכיר בלבד
    if not check_is_employee_only(raw_text):
        return False, "עדיין לא למדתי לנתח דוחות של מי שאיננו שכיר בלבד. אני חושב שאלמד עוד ואוכל לעשות גם את זה."

    # מסנן 5: חברת כלל
    if check_clal(raw_text):
        return False, "יש לי קושי לקרוא את הדוחות של חברת כלל. נסה שוב בקרוב. ייתכן שאתגבר על הקושי ואז אוכל לסייע לך."

    return True, ""

# ============================================================
# הקוד שמתחיל כאן נועד לחלץ את הטבלאות מקבצי ה-PDF
# ============================================================

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

def process_audit_v29(client, text):
    prompt = f"""You are a RAW TEXT TRANSCRIBER. Your ONLY job is to copy characters from the text to JSON.
    
    CRITICAL INSTRUCTIONS:
    1. ZERO INTERPRETATION: Do not flip digits (e.g., 67 remains 67). 
    2. ZERO ROUNDING: If a return is 0.17%, copy 0.17%. Do NOT round to 1.0%.
    3. TABLE E SUMMARY: 
       - The 'סה"כ' row must be mapped STRICTLY. 
       - The total of the total (the largest sum) MUST be in the 'סה"כ' column.
       - 'מועד' and 'חודש' must be empty strings.
    
    JSON STRUCTURE:
    {{
      "table_a": {{"rows": [{{"תיאור": "", "סכום בש\"ח": ""}}]}},
      "table_b": {{"rows": [{{"תיאור": "", "סכום בש\"ח": ""}}]}},
      "table_c": {{"rows": [{{"תיאור": "", "אחוז": ""}}]}},
      "table_d": {{"rows": [{{"מסלול": "", "תשואה": ""}}]}},
      "table_e": {{"rows": [{{ "שם המעסיק": "", "מועד": "", "חודש": "", "שכר": "", "עובד": "", "מעסיק": "", "פיצויים": "", "סה\"כ": "" }}]}}
    }}
    TEXT: {text}"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a mechanical OCR tool. You copy characters exactly. You do not use logic, you do not round, and you do not flip numbers."},
                  {"role": "user", "content": prompt}],
        temperature=0,  # ביטול כל "יצירתיות" או ניחושים
        response_format={"type": "json_object"}
    )
    data = json.loads(res.choices[0].message.content)

    # תיקון הסטות וחישוב שכר ב-Python (ללא AI)
    rows_e = data.get("table_e", {}).get("rows", [])
    if len(rows_e) > 1:
        last_row = rows_e[-1]

        # 1. חישוב שכר נקי
        salary_sum = sum(clean_num(r.get("שכר", 0)) for r in rows_e[:-1])

        # 2. תיקון הסטה (Shift Fix): אם הסה"כ הכללי זז ימינה לעמודת הפיצויים
        vals = [last_row.get("עובד"), last_row.get("מעסיק"), last_row.get("פיצויים"), last_row.get("סה\"כ")]
        cleaned_vals = [clean_num(v) for v in vals]
        max_val = max(cleaned_vals)

        # אם המספר הכי גדול (הסה"כ) לא נמצא בעמודת הסה"כ - נזיז הכל למקום
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

        # 3. קיבוע שכר וניקוי תאריכים
        last_row["שכר"] = f"{salary_sum:,.0f}"
        last_row["מועד"] = ""
        last_row["חודש"] = ""
        last_row["שם המעסיק"] = "סה\"כ"

    return data

# עד כאן הקוד לחילוץ הידע מהקבצים
# ============================================================

# ============================================================
# חישוב שנים לפרישה והכנסה מבוטחת
# ============================================================

def calc_nper(rate_annual, pv, fv):
    """
    חישוב מספר שנים לפרישה לפי נוסחת NPER עם PMT=0.
    rate_annual: ריבית שנתית (0.0386)
    pv: יתרת הכספים בקרן (ערך חיובי)
    fv: היעד הצבירה (ערך חיובי)
    נוסחה: n = ln(fv / pv) / ln(1 + rate)
    """
    if pv <= 0 or fv <= 0:
        return None
    try:
        n = math.log(fv / pv) / math.log(1 + rate_annual)
        return round(n, 1)
    except (ValueError, ZeroDivisionError):
        return None

def calc_years_to_retirement_and_insured_income(data):
    """
    מחשב:
    1. שנים לפרישה – NPER(3.86%, PMT=0, PV=יתרה בקרן, FV=קצבה_חודשית * 190)
    2. הכנסה מבוטחת – על בסיס שיעור ההפקדה מטבלא ה' וערך שחרור מתשלום מטבלא א'
    """
    st.subheader("📊 ניתוח פיננסי")

    # ──────────────────────────────────────────────
    # שלב 1: שנים לפרישה
    # ──────────────────────────────────────────────
    rows_a = data.get("table_a", {}).get("rows", [])
    rows_b = data.get("table_b", {}).get("rows", [])

    # ערך עתידי: השורה העליונה בטבלא א' (קצבה חודשית צפויה) * 190
    monthly_pension = clean_num(rows_a[0].get("סכום בש\"ח", 0)) if rows_a else 0.0
    fv_target = monthly_pension * 190

    # ערך נוכחי: השורה האחרונה בטבלא ב' (יתרת הכספים בסוף תקופת הדוח)
    current_balance = clean_num(rows_b[-1].get("סכום בש\"ח", 0)) if rows_b else 0.0

    years = calc_nper(0.0386, current_balance, fv_target)

    if years is not None:
        st.markdown(f'<div class="info-box">📅 שנים לפרישה (לפי תשואה שנתית של 3.86%): <b>{years}</b> שנים</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="val-error">⚠️ לא ניתן לחשב שנים לפרישה – חסרים נתונים מטבלאות א\' ו-ב\'.</div>', unsafe_allow_html=True)

    # ──────────────────────────────────────────────
    # שלב 2: הכנסה מבוטחת
    # ──────────────────────────────────────────────
    rows_e = data.get("table_e", {}).get("rows", [])

    if not rows_e:
        st.markdown('<div class="val-error">⚠️ אין נתונים בטבלא ה\' לחישוב הכנסה מבוטחת.</div>', unsafe_allow_html=True)
        return

    last_e = rows_e[-1]  # שורת הסיכום התחתונה

    # סה"כ הפקדות (הערך השני בגובהו בשורת הסיכום – אימות: עמודת סה"כ)
    total_deposits = clean_num(last_e.get("סה\"כ", 0))

    # סה"כ שכר (הערך הגבוה ביותר בשורת הסיכום – אימות: עמודת שכר)
    total_salary = clean_num(last_e.get("שכר", 0))

    # חישוב שיעור ההפקדה
    if total_salary == 0:
        st.markdown('<div class="val-error">⚠️ לא ניתן לחשב שיעור הפקדה – סה"כ שכר הוא 0.</div>', unsafe_allow_html=True)
        return

    deposit_rate = total_deposits / total_salary

    # אימות טווח שיעור ההפקדה
    if 0.185 <= deposit_rate <= 0.2283:
        st.markdown(f'<div class="val-success">✅ שיעור הפקדה: {deposit_rate*100:.2f}% (תקין – בטווח 18.5%–22.83%)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="val-error">⚠️ שיעור הפקדה: {deposit_rate*100:.2f}% – חורג מהטווח הצפוי (18.5%–22.83%). בדוק את הנתונים.</div>', unsafe_allow_html=True)

    # ערך שחרור מתשלום: השורה האחרונה בטבלא א'
    waiver_value = clean_num(rows_a[-1].get("סכום בש\"ח", 0)) if rows_a else 0.0

    # הפקדה מבוטחת = שחרור מתשלום / 0.94
    insured_deposit = waiver_value / 0.94 if waiver_value > 0 else 0.0

    # הכנסה מבוטחת = הפקדה מבוטחת / שיעור ההפקדה
    insured_income = insured_deposit / deposit_rate if deposit_rate > 0 else 0.0

    st.markdown(f"""
    <div class="info-box">
        💼 <b>ניתוח הכנסה מבוטחת:</b><br>
        • ערך שחרור מתשלום (שורה אחרונה בטבלא א'): <b>{waiver_value:,.2f} ₪</b><br>
        • הפקדה מבוטחת (שחרור / 0.94): <b>{insured_deposit:,.2f} ₪</b><br>
        • שיעור הפקדה: <b>{deposit_rate*100:.2f}%</b><br>
        • <u>הכנסה מבוטחת: <b>{insured_income:,.2f} ₪</b></u>
    </div>
    """, unsafe_allow_html=True)

# עד כאן הקוד של חישוב השנים לפרישה וההכנסה המבוטחת
# ============================================================


# ============================================================
# ממשק משתמש
# ============================================================
st.title("📋 חילוץ נתונים פנסיוני - גירסה 29.0")
client = init_client()

if client:
    file = st.file_uploader("העלה דוח PDF", type="pdf")
    if file:
        with st.spinner("מעתיק נתונים כפי שהם (ללא שיקול דעת AI)..."):
            file_bytes = file.read()
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            raw_text = "\n".join([page.get_text() for page in pdf_doc])

            # ── חילוץ ראשוני של טבלא א' לצורך הסינון בלבד ──
            # (חילוץ מהיר מטקסט לפני קריאה ל-AI, לצורך ספירת שורות)
            temp_table_a_rows = [
                line for line in raw_text.splitlines()
                if line.strip() and any(c.isdigit() for c in line)
            ]

            # הרצת 5 מסנני הסינון
            passed, filter_msg = run_filters(pdf_doc, raw_text, temp_table_a_rows)

            if not passed:
                st.error(filter_msg)
            else:
                # ── חילוץ הטבלאות מקבצי ה-PDF ──
                data = process_audit_v29(client, raw_text)

                if data:
                    # אימות הצלבה
                    perform_cross_validation(data)

                    # הצגת הטבלאות
                    display_pension_table(data.get("table_a", {}).get("rows"), "א. תשלומים צפויים", ["תיאור", "סכום בש\"ח"])
                    display_pension_table(data.get("table_b", {}).get("rows"), "ב. תנועות בקרן", ["תיאור", "סכום בש\"ח"])
                    display_pension_table(data.get("table_c", {}).get("rows"), "ג. דמי ניהול והוצאות", ["תיאור", "אחוז"])
                    display_pension_table(data.get("table_d", {}).get("rows"), "ד. מסלולי השקעה", ["מסלול", "תשואה"])
                    display_pension_table(data.get("table_e", {}).get("rows"), "ה. פירוט הפקדות", ["שם המעסיק", "מועד", "חודש", "שכר", "עובד", "מעסיק", "פיצויים", "סה\"כ"])

                    # ── חישוב שנים לפרישה והכנסה מבוטחת ──
                    calc_years_to_retirement_and_insured_income(data)
