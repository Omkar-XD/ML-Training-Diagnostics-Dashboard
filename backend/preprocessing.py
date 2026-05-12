import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def basic_eda(df):
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def handle_missing(df, strategy="fill"):

    # Drop rows with missing values
    if strategy == "drop":
        return df.dropna()

    # Fill numeric columns with median
    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns with mode
    categorical_cols = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for col in categorical_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def split_data(X, y, test_size=0.2):

    # =========================
    # REMOVE HIGH-CARDINALITY ID COLUMNS
    # =========================

    drop_cols = []

    for col in X.columns:

        if X[col].dtype == "object":

            unique_ratio = X[col].nunique() / len(X)

            # Likely ID/hash/random strings
            if unique_ratio > 0.9:
                drop_cols.append(col)

    if drop_cols:
        X = X.drop(columns=drop_cols)

    # =========================
    # ENCODE CATEGORICAL FEATURES
    # =========================

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_cols) > 0:

        X = pd.get_dummies(
            X,
            columns=categorical_cols,
            drop_first=True
        )

    # =========================
    # ENCODE TARGET IF CATEGORICAL
    # =========================

    if y.dtype == "object" or str(y.dtype) == "category":

        label_encoder = LabelEncoder()

        y = label_encoder.fit_transform(y)

    # =========================
    # TRAIN / VALIDATION SPLIT
    # =========================

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    # =========================
    # SCALE FEATURES
    # =========================

    scaler = StandardScaler()

    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_val = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )

    return X_train, X_val, y_train, y_val