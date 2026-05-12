import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def basic_eda(df):
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def handle_missing(df, strategy="fill"):

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

    # Encode categorical columns
    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns

    if len(categorical_cols) > 0:
        X = pd.get_dummies(
            X,
            columns=categorical_cols,
            drop_first=True
        )

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    # Scale numeric features
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