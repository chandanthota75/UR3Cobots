import gc
import utils
from tensorflow.keras.models import load_model

model_rb = utils.load_and_verify_model("models/rb/model_rb.keras")
print("\n")
model_gl = utils.load_and_verify_model("models/gl/model_gl.keras")
print("\n")
model_tc = utils.load_and_verify_model("models/tc/model_tc.keras")

base_path = "data/processed"
datasets = ["rb", "gl", "tc"]

loaded_data = {data: utils.load_sequences(base_path, data, "test") for data in datasets}
(ted_rb, tel_rb) = loaded_data["rb"]
(ted_gl, tel_gl) = loaded_data["gl"]
(ted_tc, tel_tc) = loaded_data["tc"]

predictions_rb = utils.get_predictions(model_rb, ted_rb, "classification")
predictions_gl = utils.get_predictions(model_gl, ted_gl, "classification")
predictions_tc = utils.get_predictions(model_tc, ted_tc, "regression")

utils.save_predictions_to_csv(predictions_rb, "results/rb/predictions_rb.csv", target_names= ["Robot Protective Stop"])
utils.save_predictions_to_csv(predictions_gl, "results/gl/predictions_gl.csv", target_names= ["Grip Lost"])
utils.save_predictions_to_csv(predictions_tc, "results/tc/predictions_tc.csv", target_names= ["Tool Current"])

metrics_rb = utils.calculate_metrics(tel_rb, predictions_rb, "classification")
utils.display_metrics(metrics_rb, "classification")
utils.save_metrics_to_csv(metrics_rb, 'results/rb/metrics_rb.csv', "classification")

metrics_gl = utils.calculate_metrics(tel_gl, predictions_gl, "classification")
utils.display_metrics(metrics_gl, "classification")
utils.save_metrics_to_csv(metrics_gl, 'results/gl/metrics_gl.csv', "classification")

metrics_tc = utils.calculate_metrics(tel_tc, predictions_tc, "regression")
utils.display_metrics(metrics_tc, "regression")
utils.save_metrics_to_csv(metrics_tc, 'results/tc/metrics_tc.csv', "regression")

gc.collect()
