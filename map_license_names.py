import pandas as pd
import os

input_dir = "./data_sampled"

# Diccionario de nombres legibles
license_map = {
    "HV0003": "Uber",
    "HV0005": "Lyft",
    "HV0002": "Via",
    "HV0004": "Curb"
}

# Procesar todos los archivos
for file in sorted(os.listdir(input_dir)):
    if not file.endswith("_reduced.parquet"):
        continue

    path = os.path.join(input_dir, file)
    print(f"🔄 Actualizando operadores en: {file}")

    df = pd.read_parquet(path)

    if "hvfhs_license_num" in df.columns:
        df["hvfhs_license_num"] = df["hvfhs_license_num"].map(license_map).fillna(df["hvfhs_license_num"])
        df.to_parquet(path, index=False)
        print(f"✅ Guardado: {file}")
    else:
        print(f"⚠️  Columna 'hvfhs_license_num' no encontrada en {file}, se omite.")
