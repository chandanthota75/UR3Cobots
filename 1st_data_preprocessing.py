import gc
import utils
import numpy as np
import pandas as pd
from tabulate import tabulate
from imblearn.over_sampling import SMOTE

data = utils.load_data("data/raw/cobot_data.xlsx", "xlsx")
print(f'Dataset loaded. Shape: {data.shape}')

data = data.drop(columns="Num")
print(f'Dataset after dropping column "Num". Shape: {data.shape}')

data.rename(columns={
    'cycle': 'Cycle',
    'Robot_ProtectiveStop': 'Robot Protective Stop', 
    'grip_lost': 'Grip Lost', 
    'Tool_current': 'Tool Current'
}, inplace=True)

print(data.columns.tolist())

def process_timestamp(df):
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"].str.strip('"'),
        format="%Y-%m-%dT%H:%M:%S.%fZ",
        errors="coerce"
    )
    dropped_rows = df["Timestamp"].isnull().sum()
    if dropped_rows > 0:
        print(f"Warning: {dropped_rows} rows dropped due to invalid timestamps.")
        df = df.dropna(subset=["Timestamp"])
    
    df = df.sort_values(by="Timestamp", ascending=True)
    df = df.reset_index(drop=True)
    df["Hour"] = df["Timestamp"].dt.hour
    df["Minute"] = df["Timestamp"].dt.minute
    df["Second"] = df["Timestamp"].dt.second
    df['Time of Day'] = (df['Hour'] * 3600) + (df['Minute'] * 60) + df['Second']
    return df.drop(columns=["Timestamp", "Hour", "Minute", "Second"])

data = process_timestamp(data)
print(f'Dataset after creating new temporal feature using column "Timestamp". Shape: {data.shape}')

features = data.columns.difference(["Robot Protective Stop", "Grip Lost", "Tool Current"]).tolist()
print("Features identified:", features)
print("Features Count:", len(features))

def generate_missing_values_table(dataset, threshold=0.5):
    try:
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError("Input dataset must be a pandas DataFrame.")

        missing_values = dataset.isnull().sum()
        missing_percentage = (missing_values / len(dataset)) * 100

        summary = pd.DataFrame({
            "Column": dataset.columns,
            "Missing Values": missing_values,
            "Missing Percentage (%)": missing_percentage
        })

        summary = summary[summary["Missing Values"] > 0].reset_index(drop=True)

        null_row_threshold = int(len(dataset.columns) * threshold)
        mostly_null_rows = (dataset.isnull().sum(axis=1) > null_row_threshold).sum()

        output = []
        if summary.empty:
            output.append("No missing values found in the dataset.")
        else:
            output.append(tabulate(summary, headers="keys", tablefmt="grid", showindex=False))

        output.append(f"\nRows with >{int(threshold * 100)}% null values: {mostly_null_rows}")

        return "\n".join(output)

    except Exception as e:
        return f"An error occurred: {e}"

print(generate_missing_values_table(data))

data["Robot Protective Stop"] = data["Robot Protective Stop"].fillna("FALSE" if (data["Grip Lost"] == "FALSE").any() else data["Robot Protective Stop"])
data["Grip Lost"] = data["Grip Lost"].fillna("FALSE" if (data["Robot Protective Stop"] == "FALSE").any() else data["Grip Lost"])

data = data.ffill().bfill()
data["Grip Lost"] = data["Grip Lost"].astype(int)

print(generate_missing_values_table(data))

utils.save_data_csv(data, "data/processed", "handling_missing.csv")

del data
del features
gc.collect()

data = utils.load_data("data/processed/handling_missing.csv", "csv")

features = data.columns.difference(["Robot Protective Stop", "Grip Lost", "Tool Current"]).tolist()
print("Features identified:", features)
print("Features Count:", len(features))

def handle_outliers(df, feat, threshold):
    try:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The input dataset must be a pandas DataFrame.")

        if not isinstance(feat, list):
            raise TypeError("The features parameter must be a list of feature names.")

        for feature in feat:
            if feature not in df.columns:
                raise ValueError(f"Feature '{feature}' is not present in the dataset.")

        for feature in feat:
            z_scores = (df[feature] - df[feature].mean()) / df[feature].std()
            df[feature] = np.where(z_scores.abs() > threshold, np.nan, df[feature])

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

data = handle_outliers(data, features, threshold=3)
print(generate_missing_values_table(data, threshold=0.3))

data = data.ffill().bfill()
print(generate_missing_values_table(data, threshold=0.3))

print(f'Dataset after removing outliers. Shape: {data.shape}')
utils.save_data_csv(data, "data/processed", "outliers_removed.csv")

del data
del features
gc.collect()

data = utils.load_data("data/processed/outliers_removed.csv", "csv")
cycle_summary = data[["Time of Day", "Cycle"]].groupby('Cycle').agg(
    cycle_time=('Time of Day', lambda x: x.max() - x.min()),
    occurrences=('Cycle', 'size')
).reset_index()

utils.save_data_csv(cycle_summary, "data/processed", "cycle_summary.csv")

data['Sin Time'] = np.sin(2 * np.pi * data["Time of Day"] / 86400)
data['Cos Time'] = np.cos(2 * np.pi * data["Time of Day"] / 86400)

data['Time Phase'] = np.arctan2(data['Sin Time'], data['Cos Time'])

data['Time Phase'] = (data['Time Phase'] + 2 * np.pi) % (2 * np.pi)

data['Cycle Time'] = data.groupby('Cycle')['Cycle'].transform('size')
data = data.drop(columns=["Time of Day", "Sin Time", "Cos Time", "Cycle"])

data['Average Temperature'] = data[[f'Temperature_J{i}' for i in range(1, 6)] + ['Temperature_T0']].mean(axis=1)
data['Gradient Temperature'] = (data[[f'Temperature_J{i}' for i in range(1, 6)] + ['Temperature_T0']].max(axis=1) - 
                                data[[f'Temperature_J{i}' for i in range(1, 6)] + ['Temperature_T0']].min(axis=1))

for i in range(6):
    data[f'Speed_Direction_J{i}'] = np.sign(data[f'Speed_J{i}'])
    data[f'Speed_J{i}'] = np.abs(data[f'Speed_J{i}'])
    
    data[f'Current_Direction_J{i}'] = np.sign(data[f'Current_J{i}'])
    data[f'Current_J{i}'] = np.abs(data[f'Current_J{i}'])

data['Load Imbalance'] = (
    data[[f'Current_J{i}' for i in range(0, 6)]].max(axis=1) - 
    data[[f'Current_J{i}' for i in range(0, 6)]].min(axis=1)) / data[[f'Current_J{i}' for i in range(0, 6)]].mean(axis=1)

print(f'Dataset after creating interaction features. Shape: {data.shape}')
features = data.columns.difference(["Robot Protective Stop", "Grip Lost", "Tool Current"]).tolist()
print("Features identified:", features)
print("Features Count:", len(features))

utils.save_data_csv(data, "data/processed", "interaction_features.csv")

del data
del features
gc.collect()

data = utils.load_data("data/processed/interaction_features.csv", "csv")

features = data.columns.difference(["Robot Protective Stop", "Grip Lost", "Tool Current"]).tolist()
print("Features identified:", features)
print("Features Count:", len(features))

new_order = [
    'Time Phase', 'Cycle Time',
    *[f'Current_J{i}' for i in range(6)],
    *[f'Current_Direction_J{i}' for i in range(6)],
    *[f'Speed_J{i}' for i in range(6)],
    *[f'Speed_Direction_J{i}' for i in range(6)],
    'Temperature_T0', *[f'Temperature_J{i}' for i in range(1, 6)],
    'Average Temperature', 'Gradient Temperature', 'Load Imbalance',
    'Robot Protective Stop', 'Grip Lost', 'Tool Current'
]

data = data[new_order]
print(f'Dataset after reordering. Shape: {data.shape}')
print(data.columns.tolist())

utils.save_data_csv(data, "data/processed", "processed_data.csv")

del data
del features
gc.collect()

data = utils.load_data("data/processed/processed_data.csv", "csv")

def add_rolling_features(df, feas, window_size):
    try:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The dataset must be a pandas DataFrame.")
        if not isinstance(feas, list) or not all(isinstance(i, str) for i in feas):
            raise ValueError("Features must be a list of column names (strings).")
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        missing_features = [feature for feature in feas if feature not in df.columns]
        if missing_features:
            raise ValueError(f"The following features are missing from the dataset: {', '.join(missing_features)}")

        rolling_features = {}
        for feature in feas:
            rolling_features[f"{feature}_rolling_mean"] = df[feature].rolling(window=window_size, min_periods=1).mean()
            rolling_features[f"{feature}_rolling_std"] = df[feature].rolling(window=window_size, min_periods=1).std()

        rolling_df = df.copy()
        for ky, value in rolling_features.items():
            rolling_df[ky] = value

        return rolling_df
    except Exception as e:
        print(f"Error: {e}")
        return None

features_rll = [
    *[f'Current_J{i}' for i in range(6)],
    *[f'Speed_J{i}' for i in range(6)], 
    'Temperature_T0', *[f'Temperature_J{i}' for i in range(1, 6)]
]

cycle_summary = utils.load_data("data/processed/cycle_summary.csv", "csv")

sequence_length = int(np.mean(cycle_summary["occurrences"]))

with open("data/processed/sequence_length.txt", "w") as file:
    file.write(str(sequence_length))

print(f"Inter Mean (average cycle interval length) of operational cycle -> {sequence_length}")

data = add_rolling_features(data, features_rll, sequence_length).ffill().bfill()
print(f'Dataset - After adding rolling features. Shape: {data.shape}')

features = data.columns.difference(["Robot Protective Stop", "Grip Lost", "Tool Current"]).tolist()
print("Features identified:", features)
print("Features Count:", len(features))

utils.save_data_csv(data, "data/processed", "rolling_features.csv")
del data
del features_rll
gc.collect()

data = utils.load_data("data/processed/rolling_features.csv", "csv")

data_rb = data[data.columns.difference(["Grip Lost", "Tool Current"]).tolist()].copy()
features_rb = data_rb.columns.difference(['Robot Protective Stop']).tolist()
target_rb = "Robot Protective Stop"

data_gl = data[data.columns.difference(["Robot Protective Stop", "Tool Current"]).tolist()].copy()
features_gl = data_gl.columns.difference(['Grip Lost']).tolist()
target_gl = "Grip Lost"

data_tc = data[data.columns.difference(["Robot Protective Stop", "Grip Lost"]).tolist()].copy()
features_tc = data_tc.columns.difference(['Tool Current']).tolist()
target_tc = "Tool Current"

trd_rb, trl_rb, vad_rb, val_rb, ted_rb, tel_rb = utils.split_data(data_rb, features_rb, target_rb)
print(f'Target variable "Robot Protective Stop" dataset - Train feature Shape: {trd_rb.shape}, Train target Shape: {trl_rb.shape}')
print(f'Target variable "Robot Protective Stop" dataset - Validation features Shape: {vad_rb.shape}, Validation target Shape: {val_rb.shape}')
print(f'Target variable "Robot Protective Stop" dataset - Test features Shape: {ted_rb.shape}, Test target Shape: {tel_rb.shape}')

trd_gl, trl_gl, vad_gl, val_gl, ted_gl, tel_gl = utils.split_data(data_gl, features_gl, target_gl)
print(f'\nTarget variable "Grip Lost" dataset - Train features Shape: {trd_gl.shape}, Train target Shape: {trl_gl.shape}')
print(f'Target variable "Grip Lost" dataset - Validation features Shape: {vad_gl.shape}, Validation target Shape: {val_gl.shape}')
print(f'Target variable "Grip Lost" dataset - Test features Shape: {ted_gl.shape}, Test target Shape: {tel_gl.shape}')

trd_tc, trl_tc, vad_tc, val_tc, ted_tc, tel_tc = utils.split_data(data_tc, features_tc, target_tc)
print(f'\nTarget variable "Tool Current" dataset - Train features Shape: {trd_tc.shape}, Train target Shape: {trl_tc.shape}')
print(f'Target variable "Tool Current" dataset - Validation features Shape: {vad_tc.shape}, Validation target Shape: {val_tc.shape}')
print(f'Target variable "Tool Current" dataset - Test features Shape: {ted_tc.shape}, Test target Shape: {tel_tc.shape}')

datasets = {
    "rb": {"train": (trd_rb, trl_rb), "valid": (vad_rb, val_rb), "test": (ted_rb, tel_rb)},
    "gl": {"train": (trd_gl, trl_gl), "valid": (vad_gl, val_gl), "test": (ted_gl, tel_gl)},
    "tc": {"train": (trd_tc, trl_tc), "valid": (vad_tc, val_tc), "test": (ted_tc, tel_tc)}
}

for key, splits in datasets.items():
    for split, (data, labels) in splits.items():
        utils.save_data_csv(data, f"data/processed/{key}/{split}", f"{split}_data_{key}.csv")
        utils.save_data_csv(labels, f"data/processed/{key}/{split}", f"{split}_labels_{key}.csv")

def smote_time_series_balancing(train_data, train_labels):
    try:
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("train_data must be a pandas DataFrame.")
        if not isinstance(train_labels, (pd.Series, np.ndarray, list)):
            raise ValueError("train_labels must be a pandas Series, numpy array, or list.")

        smote = SMOTE()
        balanced_data, balanced_labels = smote.fit_resample(train_data, train_labels)
        balanced_data = pd.DataFrame(balanced_data, columns=train_data.columns)
        balanced_data['Labels'] = balanced_labels

        if 'Time Phase' in balanced_data.columns:
            balanced_data.sort_values(by='Time Phase', inplace=True)
        else:
            raise ValueError("The dataset must contain 'Time Phase' column to sort.")

        balanced_labels = balanced_data.pop('Labels')
        return balanced_data, balanced_labels

    except Exception as e:
        raise RuntimeError(f"An error occurred during SMOTE analysis: {e}")

print("Before time series SMOTE:")
print(f"Class distribution in 'Robot Protective Stop' target variable: \n{trl_rb.value_counts()}")
print(f"\nClass distribution in 'Grip Lost' target variable: \n{trl_gl.value_counts()}")

print(f'"Robot Protective Stop" dataset - Features Shape: {trd_rb.shape}, Labels Shape: {trl_rb.shape}')
print(f'"Grip Lost" dataset - Features Shape: {trd_gl.shape}, Labels Shape: {trl_gl.shape}')

trd_rb, trl_rb = smote_time_series_balancing(trd_rb, trl_rb)
trd_gl, trl_gl = smote_time_series_balancing(trd_gl, trl_gl)

print(f'"Robot Protective Stop" dataset - Features Shape: {trd_rb.shape}, Labels Shape: {trl_rb.shape}')
print(f'"Grip Lost" dataset - Features Shape: {trd_gl.shape}, Labels Shape: {trl_gl.shape}')

print("\nAfter time series SMOTE:")
print(f"Class distribution in 'Robot Protective Stop' target variable: \n{trl_rb.value_counts()}")
print(f"\nClass distribution in 'Grip Lost' target variable: \n{trl_gl.value_counts()}")

def check_duplicates(content):
    feature_duplicates = content.duplicated().sum()

    print(f"Number of duplicate rows in features: {feature_duplicates}")
    if feature_duplicates > 0:
        print("Duplicates are present in the dataset.")
    else:
        print("No duplicates found in the dataset.")

check_duplicates(trd_rb)
check_duplicates(trd_gl)

datasets = {
    "rb": (trd_rb, trl_rb),
    "gl": (trd_gl, trl_gl)
}

for key, (data, labels) in datasets.items():
    utils.save_data_csv(data, f"data/processed/{key}/train", f"balanced_train_data_{key}.csv")
    utils.save_data_csv(labels, f"data/processed/{key}/train", f"balanced_train_labels_{key}.csv")

new_order = [
    'Time Phase', 'Cycle Time', 
    *[f'Current_J{i}' for i in range(6)],
    *[f'Current_Direction_J{i}' for i in range(6)],
    *[f'Current_J{i}_rolling_mean' for i in range(6)], 
    *[f'Current_J{i}_rolling_std' for i in range(6)], 
    *[f'Speed_J{i}' for i in range(6)],
    *[f'Speed_Direction_J{i}' for i in range(6)],
    *[f'Speed_J{i}_rolling_mean' for i in range(6)], 
    *[f'Speed_J{i}_rolling_std' for i in range(6)], 
    'Temperature_T0', *[f'Temperature_J{i}' for i in range(1, 6)], 
    'Temperature_T0_rolling_mean', *[f'Temperature_J{i}_rolling_std' for i in range(1, 6)],
    'Temperature_T0_rolling_std', *[f'Temperature_J{i}_rolling_std' for i in range(1, 6)],
    'Average Temperature', 'Gradient Temperature', 'Load Imbalance'
]

trd_rb = trd_rb[new_order]
vad_rb = vad_rb[new_order]
ted_rb = ted_rb[new_order]

trd_gl = trd_gl[new_order]
vad_gl = vad_gl[new_order]
ted_gl = ted_gl[new_order]

trd_tc = trd_tc[new_order]
vad_tc = vad_tc[new_order]
ted_tc = ted_tc[new_order]

datasets = {
    "rb": {"train": trd_rb, "valid": vad_rb, "test": ted_rb},
    "gl": {"train": trd_gl, "valid": vad_gl, "test": ted_gl},
    "tc": {"train": trd_tc, "valid": vad_tc, "test": ted_tc}
}

for key, splits in datasets.items():
    for split, data in splits.items():
        utils.save_data_csv(data, f"data/processed/{key}/{split}", f"scaled_{split}_data_{key}.csv")

with open("data/processed/sequence_length.txt", "r") as file:
    sequence_length = int(file.read().strip())

trd_rb, trl_rb = utils.create_sequences(trd_rb, trl_rb, sequence_length, "classification")
vad_rb, val_rb = utils.create_sequences(vad_rb, val_rb, sequence_length, "classification")
ted_rb, tel_rb = utils.create_sequences(ted_rb, tel_rb, sequence_length, "classification")

trd_gl, trl_gl = utils.create_sequences(trd_gl, trl_gl, sequence_length, "classification")
vad_gl, val_gl = utils.create_sequences(vad_gl, val_gl, sequence_length, "classification")
ted_gl, tel_gl = utils.create_sequences(ted_gl, tel_gl, sequence_length, "classification")

trd_tc, trl_tc = utils.create_sequences(trd_tc, trl_tc, sequence_length, "regression")
vad_tc, val_tc = utils.create_sequences(vad_tc, val_tc, sequence_length, "regression")
ted_tc, tel_tc = utils.create_sequences(ted_tc, tel_tc, sequence_length, "regression")

datasets = {
    'Robot Protective Stop': [trd_rb, trl_rb, vad_rb, val_rb, ted_rb, tel_rb],
    'Grip Lost': [trd_gl, trl_gl, vad_gl, val_gl, ted_gl, tel_gl],
    'Tool Current': [trd_tc, trl_tc, vad_tc, val_tc, ted_tc, tel_tc]
}

for label, (x_train, y_train, x_valid, y_valid, x_test, y_test) in datasets.items():
    print(f"Train data \"{label}\": {x_train.shape}, Train labels \"{label}\": {y_train.shape}")
    print(f"Valid data \"{label}\": {x_valid.shape}, Valid labels \"{label}\": {y_valid.shape}")
    print(f"Test data \"{label}\": {x_test.shape}, Test labels \"{label}\": {y_test.shape}")

arrays = {
    "rb": {"train": (trd_rb, trl_rb), "valid": (vad_rb, val_rb), "test": (ted_rb, tel_rb)},
    "gl": {"train": (trd_gl, trl_gl), "valid": (vad_gl, val_gl), "test": (ted_gl, tel_gl)},
    "tc": {"train": (trd_tc, trl_tc), "valid": (vad_tc, val_tc), "test": (ted_tc, tel_tc)}
}

for key, splits in arrays.items():
    for split, data in splits.items():
        folder = f"data/processed/{key}/{split}/sequences"
        utils.save_sequences(data, folder, f"seq_{split}_data_{key}.npy", f"seq_{split}_labels_{key}.npy")

gc.collect()
