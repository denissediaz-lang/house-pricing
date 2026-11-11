"""Genera predicciones de SalePrice en dólares a partir de datos crudos.

Uso desde la raíz del proyecto:

    python -m house_pricing.modeling.predict
    python -m house_pricing.modeling.predict --input ruta/datos.csv --output ruta/salida.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
import yaml

PROJ_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJ_ROOT / "pipeline_config.yaml"
ID_COLUMN = "Id"


def load_config(config_path=CONFIG_PATH):
    """Lee la configuración del proyecto."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_predictions(input_path, output_path, pipeline_path, target):
    """Aplica el pipeline de inferencia a un CSV crudo y guarda Id y predicción."""
    raw_data = pd.read_csv(input_path)
    inference_pipeline = joblib.load(pipeline_path)

    predictions = pd.DataFrame(
        {ID_COLUMN: raw_data[ID_COLUMN], target: inference_pipeline.predict(raw_data)}
    )
    predictions.to_csv(output_path, index=False)
    return predictions


def main():
    config = load_config()
    default_input = PROJ_ROOT / config["DATA"]["RAW_DIR"] / config["DATA"]["TEST_FILE"]
    default_output = PROJ_ROOT / config["INFERENCE"]["PREDICTIONS_PATH"]

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=default_input, help="CSV de datos crudos")
    parser.add_argument("--output", type=Path, default=default_output, help="CSV de salida")
    args = parser.parse_args()

    predictions = generate_predictions(
        args.input,
        args.output,
        PROJ_ROOT / config["INFERENCE"]["PIPELINE_PATH"],
        config["TARGET"],
    )
    print(f"{len(predictions)} predicciones guardadas en {args.output}")


if __name__ == "__main__":
    main()
