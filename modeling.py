import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

def prepare_X_y(df: pd.DataFrame, target_col: str, features=None):
    """
    بازگرداندن X, y و نگاشت‌های factorize برای categorical.
    اگر features None باشد، تمام ستون‌های بجز target انتخاب می‌شوند.
    """
    if features is None:
        features = [c for c in df.columns if c != target_col]
    X = df[features].copy()
    y = df[target_col].copy()

    encodings = {}  # {'col': array_of_categories}
    for col in X.select_dtypes(include="object").columns:
        codes, uniques = pd.factorize(X[col].astype(str))
        X[col] = codes
        encodings[col] = uniques.tolist()

    # تبدیل بقیه non-numericهای عجیب به numeric اگر ممکن
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            try:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            except:
                X[col] = X[col].astype(str).factorize()[0]

    return X, y, encodings


def auto_select_features(X: pd.DataFrame, y: pd.Series, method="mutual_info", k=5):
    """
    انتخاب k ویژگی برتر با روش mutual_info یا rf importance.
    """
    k = min(k, X.shape[1])
    if method == "mutual_info":
        try:
            selector = SelectKBest(mutual_info_regression, k=k)
            selector.fit(X, y)
            selected = X.columns[selector.get_support()].tolist()
            return selected
        except Exception:
            return X.columns.tolist()[:k]
    elif method == "rf_importance":
        try:
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X, y)
            imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
            return imp.head(k).index.tolist()
        except Exception:
            return X.columns.tolist()[:k]
    else:
        return X.columns.tolist()[:k]


def train_models(X: pd.DataFrame, y: pd.Series, test_size=0.2, random_state=42):

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01, max_iter=10000),
        "DecisionTree": DecisionTreeRegressor(max_depth=5, random_state=42)
    }

    results = {}
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # cross val r2 (try/except because some fits may fail)
            try:
                cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
                cv_r2 = np.nanmean(cv_scores)
            except Exception:
                cv_r2 = np.nan

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            results[name] = {
                "model": model,
                "y_test": y_test.reset_index(drop=True),
                "y_pred": pd.Series(y_pred),
                "r2": r2,
                "cv_r2": cv_r2,
                "mae": mae,
                "rmse": rmse
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    # انتخاب بهترین مدل: اول مرتب بر اساس cv_r2 (بیشتر بهتر)، سپس از بین نزدیک‌ها مدل با gap کوچکتر انتخاب می‌شود
    eval_rows = []
    for name, info in results.items():
        if "error" in info:
            eval_rows.append({"model": name, "r2": -np.inf, "cv_r2": np.nan, "gap": np.inf})
        else:
            r2 = info.get("r2", -np.inf)
            cv_r2 = info.get("cv_r2", np.nan)
            gap = abs(r2 - cv_r2) if not np.isnan(cv_r2) else 0
            eval_rows.append({"model": name, "r2": r2, "cv_r2": cv_r2, "gap": gap})
    eval_df = pd.DataFrame(eval_rows).sort_values(by=["cv_r2", "gap"], ascending=[False, True])

    best_model_name = eval_df.iloc[0]["model"] if not eval_df.empty else None
    return results, best_model_name


def get_feature_importance(model, X, y):

    results = []

    # اگر مدل خطی باشد → coef_
    if hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        for i, col in enumerate(X.columns):

            # رگرسیون تک‌متغیره برای R2
            Xi = X[[col]]
            lr = LinearRegression()
            lr.fit(Xi, y)
            y_pred = lr.predict(Xi)
            simple_r2 = r2_score(y, y_pred)

            results.append({
                "Feature": col,
                "Coefficient": coefs[i],
                "Simple_R2": simple_r2
            })

    # اگر مدل درخت تصمیم باشد → feature_importances_
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        for col, imp in zip(X.columns, importances):

            # R2 ساده تک‌متغیره
            Xi = X[[col]]
            lr = LinearRegression()
            lr.fit(Xi, y)
            y_pred = lr.predict(Xi)
            simple_r2 = r2_score(y, y_pred)

            results.append({
                "Feature": col,
                "Coefficient": imp,
                "Simple_R2": simple_r2
            })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Simple_R2", ascending=False)
    return df


def get_linear_equation(model, features):
    """
    اگر مدل ضریب داشته باشد معادله را می‌سازد.
    """
    if not hasattr(model, "coef_"):
        return None, None

    coefs = np.ravel(model.coef_)
    intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0
    terms = [f"({coef:.4f}*{feat})" for coef, feat in zip(coefs, features)]
    equation = " + ".join(terms) + f" + ({intercept:.4f})"
    return equation, intercept
