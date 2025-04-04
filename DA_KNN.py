import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preparar_datos_knn(df):
    # Codificación one-hot
    data = pd.get_dummies(df, columns=['Punto'], drop_first=True)

    # Separar X e y
    X = data.drop(columns=['clase', 'Date'])
    y = data['clase']

    # División en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Escalado con MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Volver a DataFrame para legibilidad
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled_df, X_test_scaled_df, y_train, y_test