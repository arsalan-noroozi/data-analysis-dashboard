import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def plot_histogram(df, col, show_kde=True):

    # حذف پرت‌ها با IQR
    raw = df[col]
    numeric = pd.to_numeric(raw, errors="coerce").dropna()

    if numeric.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f"No numeric data available in column: {col}", ha="center", va="center")
        ax.set_axis_off()
        return fig

    if len(numeric) >= 2:
        Q1, Q3 = np.percentile(numeric, [25, 75])
    else:
        Q1 = Q3 = numeric.median()

    data = numeric
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    filtered = data[(data >= lower) & (data <= upper)]

    #  تعداد bins به روش Freedman–Diaconis
    if len(filtered) > 1:
        IQR_f = np.subtract(*np.percentile(filtered, [75, 25]))
        bin_width = 2 * IQR_f * len(filtered) ** (-1 / 3)
        bins = max(10, int((filtered.max() - filtered.min()) / bin_width))
    else:
        bins = 10

    # ---  ساخت نمودار ---
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("whitegrid")

    sns.histplot(
        filtered,
        bins=bins,
        kde=show_kde,
        color="#1f77b4",
        alpha=0.85,
        edgecolor="#333333",
        linewidth=0.6,
        ax=ax
    )

    #خطوط میانگین و میانه
    mean_val = numeric.mean()
    median_val = numeric.median()

    ax.axvline(mean_val, color="#d62728", linestyle="--", linewidth=1.8,
               label=f"Mean: {mean_val:.2f}")
    ax.axvline(median_val, color="#2ca02c", linestyle="-.", linewidth=1.8,
               label=f"Median: {median_val:.2f}")

    ax.set_title(f"Histogram of {col}", fontsize=15, fontweight='bold', color="#333333")
    ax.set_xlabel(col, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(axis='y', linestyle=":", alpha=0.5)
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    return fig


def plot_boxplot(df, col):

    # اطمینان از عددی بودن داده
    raw = df[col]
    data = pd.to_numeric(raw, errors="coerce").dropna()

    if data.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, f"No numeric data available in column: {col}", ha="center", va="center")
        ax.set_axis_off()
        return fig

    if len(data) >= 2:
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
    else:
        Q1 = Q3 = data.median()

    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR

    # شمارش واقعی outlierها بر اساس همین بازه
    outliers = data[(data < lower_whisker) | (data > upper_whisker)]
    n_outliers = len(outliers)

    mean_val = data.mean()
    median_val = np.median(data)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_style("whitegrid")

    sns.boxplot(
        x=data,
        ax=ax,
        color="#ff7f0e",
        linewidth=1.5,
        showcaps=True,
        boxprops=dict(alpha=0.9, facecolor="#ffbb78"),  # رنگ داخلی
        medianprops=dict(color="black", linewidth=2.0),  # رنگ خط میانه
        showfliers=False
    )

    # رسم میانگین (خط عمودی / خطی) و میانه بصورت نقطه
    ax.axvline(mean_val, color="#9467bd", linestyle="--", linewidth=1.8,
               label=f"Mean: {mean_val:.2f}")
    ax.scatter([median_val], [0], color="white", edgecolor="black", zorder=5, s=80, label=f"Median: {median_val:.2f}",
               linewidth=1.5)  # نقطه میانه

    # توضیحات و عنوان (شمار outlier)
    ax.set_title(f"Boxplot of {col} — Outliers hidden (count = {n_outliers})", fontsize=15, fontweight="bold",
                 color="#333333")
    ax.set_xlabel(col, fontsize=12)
    ax.set_yticks([])  # حذف برچسب‌های عمودی

    # حاشیه‌نویسی (Annotation)
    ax.annotate(f"Q1={Q1:.2f}\nMedian={median_val:.2f}\nQ3={Q3:.2f}\nIQR={IQR:.2f}",
                xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", alpha=0.8, edgecolor="#666666"))

    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    # *تنظیم استایل و اندازه*
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.set_style("white")

    corr = df.corr(numeric_only=True)

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=1.0,  # خطوط ضخیم‌تر بین خانه‌ها
        linecolor='white',  # رنگ خطوط سفید
        cbar_kws={'shrink': 0.8},  # نوار رنگی کوچکتر
        ax=ax
    )
    ax.set_title("Correlation Heatmap", fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig


def plot_group_bar(df, group_col, target_col, top_n=20):
    """Barplot میانگین target_col بر اساس گروه group_col، با ادغام سایر گروه‌ها در 'Other'"""

    # محاسبه میانگین هر گروه
    grouped = df.groupby(group_col, dropna=False)[target_col].mean().sort_values(ascending=False)

    # بررسی تعداد گروه‌ها
    if len(grouped) > top_n:
        # جدا کردن top_n گروه اول
        top_groups = grouped.head(top_n)

        # محاسبه میانگین بقیه گروه‌ها
        other_mean = grouped.iloc[top_n:].mean()

        # افزودن گروه "Other"
        grouped = pd.concat([top_groups, pd.Series({"Other": other_mean})])

    # رسم نمودار
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=grouped.index, y=grouped.values, ax=ax, palette="coolwarm")

    ax.set_title(f"Average {target_col} by {group_col}", fontweight='bold')
    ax.set_xlabel(group_col)
    ax.set_ylabel(f"Mean of {target_col}")

    # چرخاندن برچسب‌ها برای وضوح
    plt.xticks(rotation=75)
    plt.tight_layout()

    return fig


def plot_group_pie(df, group_col, target_col, top_n=50, min_percent=1.0):
    """
    نمودار دایره‌ای مجموع target_col بر اساس group_col.
    گروه‌های با سهم کمتر از min_percent یا بیش از top_n در 'Other' ادغام می‌شوند.
    """
    # حذف ردیف‌های خالی
    temp_df = df[[group_col, target_col]].dropna()

    # محاسبه مجموع برای هر گروه
    grouped = (
        temp_df.groupby(group_col, dropna=False)[target_col]
        .sum()
        .sort_values(ascending=False)
    )

    # محاسبه درصد هر گروه
    total = grouped.sum()
    percent = (grouped / total) * 100

    # مرحله ۱: جدا کردن گروه‌های کم‌سهم و محاسبه مجموعشان
    small_groups = grouped[percent < min_percent]
    grouped = grouped[percent >= min_percent]

    # مرحله ۲: محدود کردن به top_n و ذخیره بقیه برای Other
    if len(grouped) > top_n:
        top_groups = grouped.head(top_n)
        others_extra = grouped.iloc[top_n:]
        grouped = top_groups
    else:
        others_extra = pd.Series(dtype=float)

    # ترکیب همه‌ی باقی‌مانده‌ها (زیر ۱٪ + مازاد بر top_n)
    other_total = pd.concat([small_groups, others_extra]).sum()
    if other_total > 0:
        grouped = pd.concat([grouped, pd.Series({"Other": other_total})])

    # رسم نمودار دایره‌ای
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        grouped.values,
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": 7},
    )

    # تنظیم زاویه و موقعیت درصدها
    for i, autotext in enumerate(autotexts):
        ang = (wedges[i].theta2 + wedges[i].theta1) / 2
        rotation = ang if np.cos(np.deg2rad(ang)) >= 0 else ang + 180
        autotext.set_rotation(rotation)
        autotext.set_rotation_mode("anchor")

    # افزودن برچسب‌های گروه با چرخش در راستای دایره
    for i, label in enumerate(grouped.index):
        ang = (wedges[i].theta2 + wedges[i].theta1) / 2
        x, y = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        rotation = ang if x >= 0 else ang + 180
        alignment = "left" if x >= 0 else "right"

        ax.text(
            1.02 * x, 1.02 * y, label,
            ha=alignment, va="center",
            rotation=rotation,
            rotation_mode="anchor",
            fontsize=7
        )

    ax.set_title(f"Percentage of {target_col} by {group_col}", fontsize=12, pad=20, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_actual_vs_pred(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, y_pred, alpha=0.6)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted", fontweight='bold')
    ax.grid(True)
    return fig


def plot_regression_scatter_line(X, y, model, feature):
    """
    رسم scatter y vs selected feature
    و رسم خط رگرسیون تک‌متغیره واقعی (نه مدل چندمتغیره)
    """
    target_name = y.name

    # --- مرحله 1: scatter ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X[feature], y, alpha=0.6, label="Actual")

    # --- مرحله 2: مدل تک‌متغیره واقعی برای رسم خط ---

    Xi = X[[feature]]
    lr_single = LinearRegression()
    lr_single.fit(Xi, y)

    # مقدارهای مرتب شده برای رسم خط
    xs = np.linspace(X[feature].min(), X[feature].max(), 200).reshape(-1, 1)
    ys = lr_single.predict(xs)

    # رسم خط رگرسیون درست
    ax.plot(xs, ys, color="red", linewidth=2, label="Best Fit Line")

    ax.set_xlabel(feature)
    ax.set_ylabel(target_name)
    ax.legend()
    ax.grid(True)
    ax.set_title(f"Regression: {feature} vs {target_name}", fontweight='bold')

    return fig
