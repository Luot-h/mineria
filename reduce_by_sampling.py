import pandas as pd
import os

input_dir = "./raw-data"
output_dir = "./data_sampled"
os.makedirs(output_dir, exist_ok=True)

# Procesar cada archivo
for file in os.listdir(input_dir):
    if file.endswith(".parquet") and "2024 -" in file:
        path = os.path.join(input_dir, file)
        print(f"🔄 Leyendo: {file}")

        df = pd.read_parquet(path)
        print(f"📊 Dataset original: {len(df):,} filas")

        # Validar que existe la columna de empresa
        if "hvfhs_license_num" not in df.columns:
            print(f"⚠️  No se encontró columna 'hvfhs_license_num' en {file}, saltando...")
            continue

        # Muestreo estratificado: 5% por empresa
        df_sampled = df.groupby("hvfhs_license_num", group_keys=False).apply(
            lambda x: x.sample(frac=0.05, random_state=42)
        )

        output_file = file.replace(".parquet", "_reduced.parquet")
        output_path = os.path.join(output_dir, output_file)

        df_sampled.to_parquet(output_path, index=False)
        print(f"✅ Guardado: {output_path} ({len(df_sampled):,} filas)")
