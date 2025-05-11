import pandas as pd
import os

# Rutas base
input_dir = "./raw-data"
output_dir = "./data_sampled"
os.makedirs(output_dir, exist_ok=True)

# Cargar zonas
zones = pd.read_csv("C:\\Users\\diego\\Workspace\\Escuela\\mineria\\data\\taxi_zone_lookup.csv")
zones = zones.rename(columns={"LocationID": "PULocationID", "Zone": "pickup_zone"})
zones["PULocationID"] = zones["PULocationID"].astype("int32")

# Recorrer archivos
for file in sorted(os.listdir(input_dir)):
    if not file.endswith(".parquet") or "2024 -" not in file:
        continue

    print(f"🔄 Procesando: {file}")
    path = os.path.join(input_dir, file)
    df = pd.read_parquet(path)

    # Muestreo estratificado
    if "hvfhs_license_num" in df.columns:
        df = df.groupby("hvfhs_license_num", group_keys=False).apply(
            lambda x: x.sample(frac=0.05, random_state=42)
        )

    # Merge con zonas
    if "PULocationID" in df.columns:
        df["PULocationID"] = df["PULocationID"].astype("int32")
        df = df.merge(zones[["PULocationID", "pickup_zone"]], on="PULocationID", how="left")

    # Procesar datetime
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["pickup_hour"] = df["pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday
    df["pickup_month"] = df["pickup_datetime"].dt.month

    # Guardar
    output_path = os.path.join(output_dir, file.replace(".parquet", "_reduced.parquet"))
    df.to_parquet(output_path, index=False)
    print(f"✅ Guardado: {output_path} ({len(df):,} filas)")
