import json
import onnx
import onnxmltools
import xgboost as xgb
import argparse
import os
import shutil  # Import the shutil module for file copying
from onnxmltools.convert.common.data_types import FloatTensorType

def convert_xgboost_to_onnx(xgboost_model, onnx_path):
    # Convert the XGBoost model to ONNX
    initial_type = [('X', FloatTensorType([None, xgboost_model.num_features()]))]
    print("Initial type: ", initial_type)
    onnx_model = onnxmltools.convert.convert_xgboost(xgboost_model, initial_types=initial_type)
    
    # Save the ONNX model to file
    onnxmltools.utils.save_model(onnx_model, onnx_path)

def load_xgboost_model_from_json(json_model_path):
    # Load the XGBoost model from JSON
    xgboost_model = xgb.Booster(model_file=json_model_path)
    print('Model type ', getattr(xgboost_model, "operator_name", None))
    return xgboost_model

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--xgboostDir', help='Directory path for XGBoost model')
parser.add_argument('--onnxDir', help='Directory path for ONNX model')
args = parser.parse_args()

# Access the value of 'xgboostDir' option
xgboost_dir = args.xgboostDir

for file in os.listdir(xgboost_dir):
    if file.endswith(".json"):
        print("Converting " + file + " to ONNX")
        xgboost_model = load_xgboost_model_from_json(os.path.join(xgboost_dir, file))
        convert_xgboost_to_onnx(xgboost_model, os.path.join(args.onnxDir, file.replace(".json", ".onnx")))
        
        # Copy the XGBoost model to the ONNX directory
        shutil.copy(os.path.join(xgboost_dir, file + ".test.sampled.csv"), args.onnxDir)
        

