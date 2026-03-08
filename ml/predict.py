import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_FILE    = "catboost_model.pkl"
FEATURES_FILE = "catboost_selected_features.csv"
INPUT_FILE    = "data.csv"
OUTPUT_FILE   = "predictions.csv"


def predict_batch(
    input_file=INPUT_FILE,
    output_file=OUTPUT_FILE,
    model_file=MODEL_FILE,
    features_file=FEATURES_FILE,
):
    """
    Run batch predictions on a dataset and save results to CSV.

    Loads the trained CatBoost model, filters the input dataset to the
    selected features, generates predictions, and appends them as a new
    column in the output file.

    Parameters
    ----------
    input_file    : str — Path to input CSV (semicolon-separated, comma decimal)
    output_file   : str — Path to save predictions CSV
    model_file    : str — Path to serialized CatBoost model (.pkl)
    features_file : str — Path to selected features CSV (semicolon-separated)

    Output
    ------
    CSV file with all original columns plus a 'Predicted_Fss' column.
    """
    # Load model and selected features
    model = joblib.load(model_file)
    selected_features = (
        pd.read_csv(features_file, sep=";")["selected_feature"].tolist()
    )

    # Load input data and filter to selected features
    data = pd.read_csv(input_file, sep=";", decimal=",", engine="python")
    data_filtered = data[selected_features]

    # Generate predictions
    predictions = model.predict(data_filtered)

    # Append predictions and save
    data["Predicted_Fss"] = predictions
    data.to_csv(output_file, index=False, sep=";", decimal=",")

    print(f"Batch prediction complete: {len(predictions)} scenarios predicted.")
    print(f"Results saved to '{output_file}'.")


if __name__ == "__main__":
    predict_batch()
