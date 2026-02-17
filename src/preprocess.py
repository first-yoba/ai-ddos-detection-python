from sklearn.preprocessing import StandardScaler

def split_features_label(df, label_column):
    X = df.drop(label_column, axis=1)
    y = df[label_column]
    return X, y

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
