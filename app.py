import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from analysis import (
    load_data,
    clean_data,
    descriptive_stats,
    correlation_analysis,
    detect_outliers,
)
from visualization import (
    plot_histogram,
    plot_boxplot,
    plot_correlation_heatmap,
    plot_group_bar,
    plot_group_pie,
    plot_actual_vs_pred,
    plot_regression_scatter_line,
)
import modeling

# -----------------------------
# تنظیمات صفحه و تم
st.set_page_config(
    page_title="📊 ابزار تحلیل داده",
    layout="wide",
    initial_sidebar_state="auto"
)

# هدر و جداکننده
st.title("📊 پلتفرم تحلیل داده")
st.markdown("---")  # جداکننده بصری

uploaded_file = st.file_uploader(
    "📂 فایل خود را بارگذاری کنید (CSV، Excel، JSON، XML)",
    type=["csv", "xlsx", "xls", "json", "xml"]
)

if uploaded_file:
    df = load_data(uploaded_file)
    df = clean_data(df)

    # استفاده از st.metric برای نمایش اطلاعات اولیه در یک ردیف
    col_a, col_b = st.columns(2)
    col_a.metric("ردیف‌ها (مشاهدات)", df.shape[0])
    col_b.metric("ستون‌ها (ویژگی‌ها)", df.shape[1])

    # تب‌بندی
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 داده‌ها", "📈 توصیفی", "🌡️ همبستگی", "🖼️ مصورسازی", "🚧 پرت‌ها", "🧠 مدل سازی"
    ])

    # -----------------------------
    with tab1:
        st.header("📋 پیش‌نمایش داده‌ها")
        st.markdown("---")

        # تبدیل ستون‌ها جهت جلوگیری از مشکل sort
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].astype(float)
            else:
                try:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="ignore")
                except:
                    pass

        df_display = df.copy()
        df_display.insert(0, ".No", range(1, len(df_display) + 1))

        # استفاده از AgGrid
        gb = GridOptionsBuilder.from_dataframe(df_display)
        gb.configure_default_column(sortable=True, filter=True, resizable=True, autoHeight=True)
        gb.configure_column(".No", header_name=".No", editable=False, valueGetter="node.rowIndex + 1", width=80,
                            cellStyle={"backgroundColor": "#f0f2f6"})
        gridOptions = gb.build()
        AgGrid(
            df_display,
            gridOptions=gridOptions,
            enable_enterprise_modules=False,
            height=400,
            allow_unsafe_jscode=True,
            theme="streamlit"
        )

    # -----------------------------
    with tab2:
        st.header("📈 آمار توصیفی")
        st.markdown("---")

        numeric_stats, categorical_stats = descriptive_stats(df)

        st.subheader("ستون‌های عددی (Numeric):")
        st.dataframe(numeric_stats, use_container_width=True)

        st.subheader("ستون‌های متنی (Categorical):")
        st.dataframe(categorical_stats, use_container_width=True)

    # -----------------------------
    with tab3:
        st.header("🌡️ ماتریس همبستگی")
        st.markdown("---")

        try:
            corr = correlation_analysis(df)
            if corr.empty:
                st.warning("هیچ ستون عددی برای محاسبه همبستگی یافت نشد.")
            else:
                st.subheader("ماتریس ضریب همبستگی:")
                st.dataframe(corr.style.background_gradient(cmap='coolwarm', axis=None).format(precision=3),
                             use_container_width=True)

                st.subheader("نقشه حرارتی (Heatmap):")
                st.pyplot(plot_correlation_heatmap(df))
        except Exception as e:
            st.error(f"خطا در محاسبه همبستگی: {e}")

    # -----------------------------
    with tab4:
        st.header("🖼️ مصورسازی داده‌ها")
        st.markdown("---")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        if numeric_cols:
            st.subheader("توزیع (Distribution) یک متغیر عددی")
            col = st.selectbox("📍 انتخاب ستون عددی:", numeric_cols, key="viz_numeric_col")

            c1, c2 = st.columns(2)
            with c1:
                st.caption("نمودار هیستوگرام و تخمین چگالی (KDE)")
                st.pyplot(plot_histogram(df, col))
            with c2:
                st.caption("نمودار Boxplot (با پنهان‌سازی پرت‌ها)")
                st.pyplot(plot_boxplot(df, col))
            st.markdown("---")

        if numeric_cols and categorical_cols:
            st.subheader("مقایسه مقادیر عددی بر اساس گروه‌های متنی")

            # استفاده از ستون‌های کنترلی برای ورودی‌ها
            control_col, chart_col = st.columns([1, 2])
            with control_col:
                group_col = st.selectbox("گروه‌بندی بر اساس ستون:", categorical_cols, key="group_col_viz")
                target_col = st.selectbox("ستون عددی مورد نظر:", numeric_cols, key="target_col_viz")
                chart_type = st.radio("نوع نمودار:", ["Bar Chart (میانگین)", "Pie Chart (مجموع)"])

            with chart_col:
                if chart_type == "Bar Chart (میانگین)":
                    st.caption(f"میانگین {target_col} بر اساس {group_col}")
                    st.pyplot(plot_group_bar(df, group_col, target_col))
                else:
                    st.caption(f"سهم {target_col} بر اساس {group_col}")
                    st.pyplot(plot_group_pie(df, group_col, target_col))

    # -----------------------------
    with tab5:
        st.header("🚧 شناسایی داده‌های پرت")
        st.markdown("---")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            outlier_col = st.selectbox("📍 انتخاب ستون عددی:", numeric_cols, key="outlier_col_tab5")
            try:
                outliers = detect_outliers(df, outlier_col)

                st.info(f"تعداد داده پرت شناسایی‌شده: **{len(outliers)}**")

                if len(outliers) > 0:
                    st.markdown("**لیست داده‌های پرت:**")

                    def highlight_outlier_col(s, col):
                        """تابع کمکی برای هایلایت کردن ستون داده پرت."""
                        # برای ستون مورد نظر رنگ زرد، برای بقیه بدون رنگ
                        return ['background-color: yellow' if s.name == col else '' for v in s]

                    st.dataframe(
                        outliers.style.apply(
                            highlight_outlier_col,
                            col=outlier_col,
                            axis=1
                        ),
                        use_container_width=True
                    )
                else:
                    st.success("هیچ داده پرت شناسایی نشد. داده‌ها تمیز هستند. ✅")
            except Exception as e:
                st.error(f"خطا در شناسایی داده‌های پرت: {e}")

    # -----------------------------
    with tab6:
        st.header("🧠 مدل سازی رگرسیون")
        st.markdown("---")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if len(numeric_cols) < 2:
            st.warning("برای رگرسیون حداقل ۲ ستون عددی لازم است.")
        else:
            # گروه‌بندی ورودی‌ها در یک expander
            with st.expander("🛠️ تنظیمات مدل سازی (Target و Features)", expanded=True):
                target = st.selectbox("🎯 انتخاب ستون هدف (Target):", numeric_cols, key="target_widget")

                st.markdown("### ⚙️ انتخاب متغیرهای مستقل (Features)")
                feature_mode = st.radio("روش انتخاب ویژگی:", ["all", "manual", "auto"], horizontal=True,
                                        key="feature_mode")

                manual_features = []
                auto_method = "mutual_info"
                auto_k = 5

                if feature_mode == "manual":
                    manual_features = st.multiselect(
                        "انتخاب دستی متغیرها (چندتایی قابل انتخاب)",
                        [c for c in df.columns if c != target],
                        key="manual_features"
                    )
                elif feature_mode == "auto":
                    auto_col1, auto_col2 = st.columns(2)
                    with auto_col1:
                        auto_method = st.selectbox("روش خودکار انتخاب ویژگی:", ["mutual_info", "rf_importance"],
                                                   key="auto_method")
                    with auto_col2:
                        auto_k = st.number_input("تعداد ویژگی انتخابی (k):", min_value=1, max_value=30, value=5,
                                                 key="auto_k")

            if st.button("🚀 آموزش مدل‌ها و انتخاب بهترین مدل", type="primary", use_container_width=True,
                         key="train_models_btn"):
                with st.spinner("در حال آماده‌سازی داده‌ها و آموزش مدل‌ها..."):
                    # تعیین features نهایی
                    if feature_mode == "all":
                        features = [c for c in df.columns if c != target]
                    elif feature_mode == "manual":
                        features = manual_features
                    else:
                        X_tmp, y_tmp, _ = modeling.prepare_X_y(df, target)
                        features = modeling.auto_select_features(X_tmp, y_tmp, method=auto_method, k=auto_k)

                    if not features:
                        st.error("هیچ فیچری انتخاب نشده است.")
                    else:
                        # ... (منطق اصلی آموزش مدل‌ها) ...
                        X, y, encodes = modeling.prepare_X_y(df, target, features)
                        results, best_model_name = modeling.train_models(X, y)
                        st.session_state["ml_results"] = results
                        st.session_state["ml_best_name"] = best_model_name
                        st.session_state["ml_features"] = features
                        st.session_state["ml_encodings"] = encodes
                        st.session_state["ml_target"] = target

            # نمایش نتایج پس از آموزش
            if "ml_results" in st.session_state:
                results = st.session_state["ml_results"]
                best_name = st.session_state["ml_best_name"]
                features = st.session_state["ml_features"]

                # جدول مقایسه مدل‌ها
                st.markdown("### 📊 مقایسه معیارها (Metrics)")

                metrics_rows = []
                for m, info in results.items():
                    if "error" in info:
                        metrics_rows.append(
                            {"Model": m, "R2": None, "R2_CV": None, "MAE": None, "RMSE": None, "Error": info["error"]})
                    else:
                        metrics_rows.append({
                            "Model": m,
                            "R2": info.get("r2"),
                            "R2_CV": info.get("cv_r2"),
                            "MAE": info.get("mae"),
                            "RMSE": info.get("rmse"),
                            "Error": None
                        })
                metrics_df = pd.DataFrame(metrics_rows).set_index("Model")

                # تابع هایلایت بهترین مدل (R2 بالاترین یا RMSE/MAE پایین‌ترین)
                def highlight_best(s):
                    if s.name in ['R2', 'R2_CV']:
                        is_best = s == s.max()
                        return ['background-color: #d4edda' if v else '' for v in is_best]
                    elif s.name in ['MAE', 'RMSE']:
                        is_best = s == s.min()
                        return ['background-color: #f8d7da' if v and v > 0 else '' for v in is_best]
                    return ['' for _ in s]


                st.dataframe(
                    metrics_df.round(4).style.apply(highlight_best, axis=0),
                    use_container_width=True
                )

                st.success(f"✅ بهترین مدل انتخاب‌شده: **{best_name}** - دارای بهترین معیار R2_CV است.")
                st.markdown("---")

                # نمایش ضرایب و نمودارها در columns
                col_chart1, col_chart2 = st.columns(2)
                best_info = results.get(best_name, {})

                with col_chart1:
                    # نمودار Actual vs Predicted
                    try:
                        st.subheader("📈 Actual vs Predicted")
                        fig = plot_actual_vs_pred(best_info["y_test"], best_info["y_pred"])
                        st.pyplot(fig)
                    except Exception:
                        st.warning("نمودار Actual vs Predicted قابل رسم نیست.")

                with col_chart2:
                    # نمایش اهمیت ویژگی‌ها/ضرایب
                    if "model" in best_info:
                        X_all, y_all, _ = modeling.prepare_X_y(df, st.session_state["ml_target"], features)
                        fi_df = modeling.get_feature_importance(best_info["model"], X_all, y_all)
                        st.subheader("🌟 اهمیت ویژگی‌ها / ضرایب")
                        st.dataframe(fi_df.reset_index(drop=True), use_container_width=True)

                st.markdown("---")

                # معادله خطی (همیشه از مدل Linear)
                if "Linear" in results and "model" in results["Linear"]:
                    st.subheader("✍️ معادله خطی (Linear Regression Equation)")
                    linear_model = results["Linear"]["model"]
                    equation, intercept = modeling.get_linear_equation(linear_model, features)

                    if equation:
                        st.code(f"y = {equation}", language='python')
                    else:
                        st.info("نتوانستم معادله خطی را بسازم.")

                st.markdown("---")

                # نمودار رگرسیون
                if len(features) >= 1:
                    st.subheader("📉 نمودار رگرسیون (Scatter + Line) — برای یک فیچر")

                    reg_col1, reg_col2 = st.columns([1, 1])

                    with reg_col1:
                        chosen_plot_feat = st.selectbox("انتخاب فیچر برای رسم نمودار (یک فیچر)", features,
                                                        key="plot_feat")

                        X_all, y_all, _ = modeling.prepare_X_y(df, st.session_state["ml_target"], features)
                        try:
                            fig2 = plot_regression_scatter_line(X_all, y_all, best_info["model"], chosen_plot_feat)
                            st.pyplot(fig2)
                        except Exception as e:
                            st.error(f"خطا در رسم نمودار رگرسیون: {e}")

                    with reg_col2:
                        st.markdown(" ")

                st.markdown("---")

                # فرم پیش‌بینی در یک ستون جانبی
                st.subheader("🔮 پیش‌بینی با مدل انتخاب‌شده")

                input_cols = st.columns(len(features) if len(features) < 4 else 4)  # حداکثر 4 ستون برای ورودی‌ها
                input_vals = {}

                for i, feat in enumerate(features):
                    with input_cols[i % len(input_cols)]:
                        if feat in df.select_dtypes(include="number").columns:
                            input_vals[feat] = st.number_input(
                                f"مقدار {feat}",
                                value=float(df[feat].mean()),
                                key=f"pred_{feat}_input"
                            )
                        else:
                            enc = st.session_state.get("ml_encodings", {}).get(feat)
                            if enc:
                                input_vals[feat] = st.selectbox(f"مقدار {feat}", options=enc, key=f"pred_{feat}_select")
                            else:
                                vals = df[feat].astype(str).unique()[:20].tolist()
                                input_vals[feat] = st.selectbox(f"مقدار {feat}", options=vals,
                                                                key=f"pred_{feat}_select_alt")

                if st.button("پیش‌بینی مقدار هدف", type="secondary", key="predict_btn"):
                    input_df = pd.DataFrame([input_vals])
                    encodings = st.session_state.get("ml_encodings", {})
                    for col in input_df.columns:
                        if col in encodings:
                            try:
                                input_df[col] = input_df[col].apply(
                                    lambda v: encodings[col].index(v) if v in encodings[col] else -1)
                            except:
                                input_df[col] = pd.factorize(input_df[col].astype(str))[0]
                        else:
                            try:
                                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0)
                            except:
                                input_df[col] = pd.factorize(input_df[col].astype(str))[0]

                    model_for_pred = st.session_state.get("ml_results", {}).get(st.session_state.get("ml_best_name"),
                                                                                {}).get("model")
                    if model_for_pred is None:
                        st.error("مدل برای پیش‌بینی موجود نیست.")
                    else:
                        try:
                            pred_val = model_for_pred.predict(input_df)[0]
                            st.success(
                                f"📌 مقدار پیش‌بینی‌شده برای **{st.session_state.get('ml_target')}**: **{pred_val:.4f}**")
                        except Exception as e:
                            st.error(f"خطا در پیش‌بینی: {e}")
