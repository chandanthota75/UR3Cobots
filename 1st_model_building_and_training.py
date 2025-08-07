import gc
import utils
import tensorflow as tf
from tensorflow.keras import models, layers, regularizers
from tensorflow.keras import callbacks

base_path = "data/processed"
datasets = ["rb", "gl", "tc"]
splits = ["train", "valid"]

loaded_data = {data: [utils.load_sequences(base_path, data, split) for split in splits] for data in datasets}
(trd_rb, trl_rb), (vad_rb, val_rb) = loaded_data["rb"]
(trd_gl, trl_gl), (vad_gl, val_gl) = loaded_data["gl"]
(trd_tc, trl_tc), (vad_tc, val_tc) = loaded_data["tc"]

def build_advanced_lstm_model(input_shape, task_type="classification"):
    if task_type not in ["classification", "regression"]:
        raise ValueError("task_type must be either 'classification' or 'regression'.")

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), dropout=0.3))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), dropout=0.3))(x)
    x = layers.BatchNormalization()(x)

    attention = layers.Attention()([x, x])
    x = layers.Concatenate()([x, attention])

    x = layers.Bidirectional(layers.LSTM(64, kernel_regularizer=regularizers.l2(0.001), dropout=0.4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)

    if task_type == "classification":
        outputs = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC()]
        )
    elif task_type == "regression":
        outputs = layers.Dense(1, activation="linear")(x)
        model = models.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.AdamW(learning_rate=lr_schedule),
            loss="log_cosh",
            metrics=["mae", tf.keras.metrics.RootMeanSquaredError()]
        )

    model.summary()
    return model

input_shape_rb = trd_rb.shape[1:]
input_shape_gl = trd_gl.shape[1:]
input_shape_tc = trd_tc.shape[1:]

print(f'Input shape for the model "Robot Protective Stop" {input_shape_rb}')
print(f'Input shape for the model "Grip Lost" {input_shape_gl}')
print(f'Input shape for the model "Tool Current" {input_shape_tc}')

model_rb = build_advanced_lstm_model(input_shape_rb)
utils.save_model_summary(model_rb, "models/rb/model_summary_rb.txt")

model_gl = build_advanced_lstm_model(input_shape_gl)
utils.save_model_summary(model_gl, "models/gl/model_summary_gl.txt")

model_tc = build_advanced_lstm_model(input_shape_tc, "regression")
utils.save_model_summary(model_tc, "models/tc/model_summary_tc.txt")

history_rb = utils.train_model(model_rb, trd_rb, trl_rb, vad_rb, val_rb, epochs=10, batch_size=32, checkpoint_filename="models/rb/best_model_rb.keras")
utils.save_model(model_rb, "models/rb/model_rb.keras")
utils.save_training_history(history_rb, "models/rb/training_history_rb.csv")

history_gl = utils.train_model(model_gl, trd_gl, trl_gl, vad_gl, val_gl, epochs=10, batch_size=32, checkpoint_filename="models/gl/best_model_gl.keras")
utils.save_model(model_gl, "models/gl/model_gl.keras")
utils.save_training_history(history_gl, "models/gl/training_history_gl.csv")

history_tc = utils.train_model(model_tc, trd_tc, trl_tc, vad_tc, val_tc, epochs=10, batch_size=32, checkpoint_filename="models/tc/best_model_tc.keras")
utils.save_model(model_tc, "models/tc/model_tc.keras")
utils.save_training_history(history_tc, "models/tc/training_history_tc.csv")

gc.collect()

