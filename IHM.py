import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
import io

st.title("Classification Studio")


def load_data(file):
    name = file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(file)

    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)  # first sheet by default

    elif name.endswith(".arff"):
        content = file.getvalue().decode("utf-8")
        data, meta = arff.loadarff(io.StringIO(content))
        df = pd.DataFrame(data)
        # ARFF text columns come back as bytes (b'value'), so decode them
        for col in df.select_dtypes([object]).columns:
            df[col] = df[col].str.decode("utf-8")
        return df


file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "arff"])

if (file is not None):
    if (st.session_state.get("file_name") != file.name):
        st.session_state["df"] = load_data(file)
        st.session_state["file_name"] = file.name

    df = st.session_state["df"]

    # ------- Display first 5 lines, shape of dataset and the type of each attribute---------
    st.subheader("First 5 lines :")
    st.dataframe(df.head())
    st.write("Shape :", df.shape)
    st.write("Attributes type :", df.dtypes)

    # ---- Fix a wrong column type ----
    st.subheader("Change a column type")
    column = st.selectbox("Column", df.columns)
    column_type = st.selectbox(
        "New type", ["int64", "float64", "str", "category"])

    if st.button("Convert"):
        try:
            if column_type in ("int64", "float64"):
                # Remove quotes and spaces, then convert to numbers
                cleaned = df[column].astype(str).str.replace(
                    '"', '').str.strip()
                converted = pd.to_numeric(cleaned, errors="coerce")

                n_failed = converted.isna().sum() - df[column].isna().sum()
                if n_failed > 0:
                    st.warning(
                        f"{n_failed} value(s) could not be converted and became NaN")
                df[column] = converted
            else:
                df[column] = df[column].astype(column_type)

            st.session_state["df"] = df
            st.success(f"'{column}' converted to {column_type}")
            st.write("New Attributes type :", df.dtypes)
        except Exception as e:
            st.error(f"Conversion failed: {e}")

    # --------------- Display the 5 values ----------
    st.subheader("5 values :")
    num_cols = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]
    cinq_valeurs = df[num_cols].describe(
    ).T[["min", "25%", "50%", "75%", "max"]]
    cinq_valeurs.columns = ["min", "Q1", "mediane", "Q3", "max"]
    st.write(cinq_valeurs)

    st.subheader("Histograms :")
    df[num_cols].hist(figsize=(14, 8), bins=30)
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # ------------ Display la matrice de correlation -------------
    st.subheader("Correlation matrix :")
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f",
                cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation matrix")
    st.pyplot(fig)
    # ------------ Delete the incorrelent column ------------
    st.subheader("Delete columns")

    cols_to_drop = st.multiselect(
        "Columns to delete :", df.columns, key="drop_cols")

    if st.button("Delete selected columns") and cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        st.session_state["df"] = df      # save the change
        st.success(f"Deleted: {cols_to_drop}. New shape: {df.shape}")

    # -------- Duplicated rows ----------
    st.subheader("Duplicated rows :")
    n_dup = df.duplicated().sum()
    st.write("Number of duplicated rows:", n_dup)
    if (n_dup > 0 and st.button("Remove duplicated rows")):
        df = df.drop_duplicates().reset_index(drop=True)
        st.session_state["df"] = df
        st.session_state["msg"] = f"{n_dup} duplicated row(s) removed. New shape: {df.shape}"
        st.rerun()

    if ("msg" in st.session_state):
        st.success(st.session_state.pop("msg"))

    # --------------- Display categorial attribute -------------
    st.subheader("Categorical attributes :")

    cat_cols = [
        c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    st.write("Categorical columns:", ", ".join(cat_cols))

    col = st.selectbox("Choose a categorical column", cat_cols, key="cat_col")

    counts = df[col].value_counts(dropna=False)

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"Distribution of **{col}**")
        st.dataframe(counts.rename("count"))
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3))
        counts.plot(kind="bar", ax=ax, edgecolor="black")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    # ------- Missing values ---------
    st.subheader("Missing values :")
    cols_with_na = df.columns[df.isna().any()].tolist()

    if (cols_with_na):
        selected = st.multiselect(
            "Columns to treat", cols_with_na, default=cols_with_na)
        method = st.selectbox(
            "Method",
            ["Drop rows", "Drop columns", "Mean",
                "Median", "Mode"]
        )

        if (st.button("Apply")):
            try:
                if (method == "Drop rows"):
                    df = df.dropna(subset=selected).reset_index(drop=True)

                elif (method == "Drop columns"):
                    df = df.drop(columns=selected)

                elif (method in ("Mean", "Median")):
                    numeric = [
                        c for c in selected if pd.api.types.is_numeric_dtype(df[c])]
                    skipped = [c for c in selected if c not in numeric]
                    for c in numeric:
                        if (method == "Mean"):
                            value = df[c].mean()
                        else:
                            value = df[c].median()
                        df[c] = df[c].fillna(value)
                    if (skipped):
                        st.warning(f"Skipped non-numeric columns: {skipped}")

                elif (method == "Mode"):
                    for c in selected:
                        df[c] = df[c].fillna(df[c].mode().iloc[0])

                st.session_state["df"] = df
                st.success(f"{method} applied. New shape: {df.shape}")
            except Exception as e:
                st.error(f"Failed: {e}")

        # Show the result after the button
    missing = df.isna().sum()
    if (missing.sum() == 0):
        st.success("No missing values left")
    else:
        st.write("Missing values per column:", missing[missing > 0])

    # ----------- Draw the box plots ----------------
    st.subheader("Box plot :")

    col = st.selectbox("Numeric column", num_cols, key="box_col")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.boxplot(df[col].dropna(), vert=False)
    ax.set_xlabel(col)
    ax.set_title(f"Box plot of {col}")
    st.pyplot(fig)

    # ------------ Treatment Outliers -------------
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    low, high = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

    outliers = df[(df[col] < low) | (df[col] > high)]
    st.write(f"Bounds: [{low:.2f}, {high:.2f}] - {len(outliers)} outlier(s)")
    st.dataframe(outliers)
