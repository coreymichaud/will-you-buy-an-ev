def create_features(df):
    # The Mode Collapse Spike
    df["is_30k_spike"] = (df["Annual_Income_USD"] == 30000.0).astype("int8")

    # The Millionaire Cliff (100% buy rate region)
    df["is_millionaire_cliff"] = (df["Annual_Income_USD"] >= 170537.0).astype("int8")

    # The Dead Zone (0% buy rate region)
    df["is_dead_zone"] = (
        (df["Annual_Income_USD"] >= 38000.0) & (df["Annual_Income_USD"] <= 42000.0)
    ).astype("int8")

    return df
