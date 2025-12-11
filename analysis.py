import pandas as pd
import numpy as np
from scipy import stats

def load_data(file) -> pd.DataFrame:
    """بارگذاری دیتاست از CSV یا Excel"""
    if file.name.endswith(".csv"):
        try:
            df = pd.read_csv(file, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file, encoding="latin1")
            except:
                df = pd.read_csv(file, encoding="windows-1252")
    elif file.name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file, engine="openpyxl")
    elif file.name.endswith(".json"):
        try:
            df = pd.read_json(file)
        except ValueError:
            file.seek(0)
            df = pd.read_json(file, lines=True)
    elif file.name.endswith(".xml"):
        df = pd.read_xml(file)
    else:
        raise ValueError("فرمت فایل پشتیبانی نمی‌شود.")
    return df


def convert_numeric_series(s: pd.Series) -> pd.Series:
    """تبدیل یک سری به عدد، مدیریت کاما و درصد"""
    s = s.astype(str).str.strip()

    s = s.str.replace("%", "", regex=False)

    if (s.str.count(",") > 1).any():
        def parse_number(val):
            val = val.replace(" ", "")

            # کاما جداکننده هزار
            try:
                return float(val.replace(",", ""))
            except:
                return val

    else:
        def parse_number(val):
            val = val.replace(" ", "")

            if val.count(",") == 1 and val.count(".") == 0:
                try:
                    return float(val.replace(",", "."))
                except:
                    return val
            else:
                try:
                    return float(val)
                except:
                    return val  # اگر عدد نبود، رشته باقی می‌ماند

    return s.map(parse_number)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """پاکسازی داده‌ها و تبدیل اعداد با کاما به float"""
    # حذف ردیف‌های کاملاً خالی
    df = df.dropna(how="all")

    # تبدیل ستون‌های object به عدد در صورت امکان
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = convert_numeric_series(df[col])

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("unknown")

    # استانداردسازی نام ستون‌ها
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    return df


def descriptive_stats(df: pd.DataFrame):
    """آمار توصیفی عددی و متنی جداگانه"""
    numeric_stats = df.describe().T
    numeric_stats["sum"] = df.select_dtypes(include=["number"]).sum()
    categorical_stats = df.describe(include=["object"]).T
    return numeric_stats, categorical_stats


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """ماتریس همبستگی (فقط ستون‌های عددی)"""
    return df.corr(numeric_only=True)


def group_comparison(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    """مقایسه آماره‌های یک ستون بر اساس گروه‌بندی"""
    if group_col not in df.columns or target_col not in df.columns:
        raise ValueError("ستون انتخاب‌شده وجود ندارد.")
    return df.groupby(group_col)[target_col].describe()


def detect_outliers(df: pd.DataFrame, col: str, threshold: float = 3) -> pd.DataFrame:
    """شناسایی داده‌های پرت بر اساس Z-score"""
    if col not in df.columns:
        raise ValueError("ستون انتخاب‌شده وجود ندارد.")
    if df[col].dtype not in [np.float64, np.int64]:
        raise TypeError("ستون انتخاب‌شده عددی نیست.")

    z_scores = np.abs(stats.zscore(df[col].dropna()))
    outliers = df.loc[z_scores > threshold]
    return outliers
