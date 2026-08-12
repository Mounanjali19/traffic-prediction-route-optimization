import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import folium

from utils.preprocess import generate_traffic_sequence
from utils.graph_utils import build_edge_index
from utils.hybrid_model import HybridGAT_LSTM


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# CPU is used for deployment because the Render instance
# does not provide a dedicated GPU.
DEVICE = "cpu"


# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "hybrid_model_final.pt"
)

EDGE_FILE = os.path.join(
    BASE_DIR,
    "data",
    "ecity_edges.csv"
)

YOLO_PATH = os.path.join(
    BASE_DIR,
    "models",
    "yolov8n.pt"
)


# ============================================================
# GLOBAL STATE
# ============================================================

# Stores the most recent traffic prediction so that
# subsequent route/map requests can reuse the same prediction.

last_prediction = None

last_prediction_metadata = None


# ============================================================
# LOAD HYBRID GAT-LSTM MODEL
# ============================================================

try:

    hybrid_model = HybridGAT_LSTM(
        in_dim=8,
        gat_hidden=64,
        gat_heads=4,
        lstm_hidden=128,
        fusion_hidden=128
    ).to(DEVICE)

    hybrid_model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        ),
        strict=True
    )

    hybrid_model.eval()

    print("Hybrid GAT-LSTM model loaded successfully.")

except Exception as e:

    print("Error loading Hybrid GAT-LSTM model:", e)

    hybrid_model = None


# ============================================================
# BUILD ROAD GRAPH
# ============================================================

try:

    SEL_EDGES, EDGE_ID_MAP, EDGE_INDEX = build_edge_index(
        EDGE_FILE
    )

    print(
        "Road graph loaded successfully.",
        "Number of selected edges:",
        len(SEL_EDGES)
    )

except Exception as e:

    print("Error building road graph:", e)

    SEL_EDGES = None
    EDGE_ID_MAP = None
    EDGE_INDEX = None


# ============================================================
# LOAD YOLO MODEL
# ============================================================

try:

    yolo_model = YOLO(YOLO_PATH)

    print("YOLO model loaded successfully.")

except Exception as e:

    print("Error loading YOLO model:", e)

    yolo_model = None


# ============================================================
# FRONTEND ROUTES
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


@app.route("/predict")
def predict_page():

    return render_template("predict.html")


@app.route("/map")
def map_page():

    return render_template("map.html")


@app.route("/upload")
def upload_page():

    return render_template("upload.html")


@app.route("/influence")
def influence_page():

    return render_template("influence.html")


# ============================================================
# TIMESTAMP HELPER
# ============================================================

def make_timestamp(date_str, time_str):

    """
    Converts date and time received from the frontend
    into a standard timestamp string.

    Example:

    date = 2026-08-12
    time = 18:30

    becomes:

    2026-08-12T18:30
    """

    if not date_str:

        date_str = datetime.now().strftime("%Y-%m-%d")

    if not time_str:

        time_str = "10:00"

    parts = str(time_str).split(":")

    hour = parts[0].zfill(2)

    minute = parts[1].zfill(2)

    return f"{date_str}T{hour}:{minute}"


# ============================================================
# TRAFFIC PREDICTION API
# ============================================================

@app.route(
    "/api/hybrid_predict",
    methods=["POST"]
)
def hybrid_predict():

    global last_prediction
    global last_prediction_metadata

    if hybrid_model is None:

        return jsonify({
            "error": "Hybrid model is not loaded."
        }), 500


    if EDGE_INDEX is None:

        return jsonify({
            "error": "Road graph is not loaded."
        }), 500


    try:

        data = request.json or {}

        date = data.get("date")

        time = data.get(
            "time",
            "10:00"
        )

        scenario = data.get(
            "scenario",
            "normal"
        )


        # Convert user input into timestamp.
        timestamp = make_timestamp(
            date,
            time
        )


        # ----------------------------------------------------
        # Generate the 12-step traffic history used by the model
        #
        # Shape before batch dimension:
        #
        # (12, N, 8)
        #
        # 12 = historical time steps
        # N  = road segments
        # 8  = traffic features
        # ----------------------------------------------------

        sequence = generate_traffic_sequence(
            timestamp,
            scenario
        )


        # Add batch dimension:
        #
        # (12, N, 8)
        #       ↓
        # (1, 12, N, 8)

        x = torch.tensor(
            sequence,
            dtype=torch.float32
        ).unsqueeze(0)


        # ----------------------------------------------------
        # MODEL INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            predictions = hybrid_model(
                x,
                EDGE_INDEX
            )

        predictions = (
            predictions
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )


        # Store the prediction so that the map and route
        # endpoints can use exactly the same prediction.

        last_prediction = predictions.copy()

        last_prediction_metadata = {

            "date": date,

            "time": time,

            "scenario": scenario,

            "timestamp": timestamp
        }


        roads = [
            f"R{i}"
            for i in range(len(predictions))
        ]


        return jsonify({

            "roads": roads,

            "speeds": predictions.tolist(),

            "unit": "km/h",

            "date": date,

            "time": time,

            "scenario": scenario

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# ROUTE RECOMMENDATION API
# ============================================================

@app.route(
    "/api/route",
    methods=["POST"]
)
def recommend_route():

    global last_prediction
    global last_prediction_metadata


    if hybrid_model is None:

        return jsonify({
            "error": "Hybrid model is not loaded."
        }), 500


    try:

        data = request.json or {}


        if "start" not in data or "end" not in data:

            return jsonify({
                "error": "Start and end road indices are required."
            }), 400


        start = int(
            data["start"]
        )

        end = int(
            data["end"]
        )


        date = data.get("date")

        time = data.get(
            "time",
            "10:00"
        )

        scenario = data.get(
            "scenario",
            "normal"
        )


        timestamp = make_timestamp(
            date,
            time
        )


        # ----------------------------------------------------
        # Reuse an existing prediction if it corresponds
        # to the same timestamp and scenario.
        # ----------------------------------------------------

        predictions = None


        if (
            last_prediction is not None
            and last_prediction_metadata is not None
        ):

            same_timestamp = (
                last_prediction_metadata["timestamp"]
                == timestamp
            )

            same_scenario = (
                last_prediction_metadata["scenario"]
                == scenario
            )


            if same_timestamp and same_scenario:

                predictions = last_prediction.copy()


        # ----------------------------------------------------
        # Otherwise generate a fresh prediction.
        # ----------------------------------------------------

        if predictions is None:

            sequence = generate_traffic_sequence(
                timestamp,
                scenario
            )


            x = torch.tensor(
                sequence,
                dtype=torch.float32
            ).unsqueeze(0)


            with torch.no_grad():

                predictions = (
                    hybrid_model(
                        x,
                        EDGE_INDEX
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .flatten()
                )


            last_prediction = predictions.copy()


            last_prediction_metadata = {

                "date": date,

                "time": time,

                "scenario": scenario,

                "timestamp": timestamp
            }


        # ----------------------------------------------------
        # Validate requested road range.
        # ----------------------------------------------------

        number_of_roads = len(predictions)


        if (
            start < 0
            or end < 0
            or start >= number_of_roads
            or end >= number_of_roads
        ):

            return jsonify({

                "error":
                f"start/end must be between "
                f"0 and {number_of_roads - 1}"

            }), 400


        # Search only within the requested corridor.

        low = min(start, end)

        high = max(start, end)


        candidate_speeds = predictions[
            low:high + 1
        ]


        if len(candidate_speeds) == 0:

            return jsonify({
                "error": "No candidate roads found."
            }), 400


        # Highest predicted speed is selected
        # as the preferred road.

        local_index = int(
            np.argmax(candidate_speeds)
        )


        recommended_index = (
            low + local_index
        )


        recommended_speed = float(
            predictions[
                recommended_index
            ]
        )


        # Convert model index back to the
        # original road edge ID.

        real_edge_id = None


        if (
            SEL_EDGES is not None
            and recommended_index < len(SEL_EDGES)
        ):

            real_edge_id = SEL_EDGES[
                recommended_index
            ]


        return jsonify({

            "start": start,

            "end": end,

            "date": date,

            "time": time,

            "scenario": scenario,

            "recommended_route_index":
                recommended_index,

            "recommended_edge_id":
                real_edge_id,

            "predicted_speed":
                recommended_speed

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# FULL TRAFFIC MAP
# ============================================================

@app.route(
    "/api/route_map_full",
    methods=["POST"]
)
def route_map_full():

    global last_prediction


    try:

        if last_prediction is None:

            return jsonify({
                "error":
                "Run traffic prediction first."
            }), 400


        speeds = last_prediction


        df = pd.read_csv(
            EDGE_FILE
        )


        # Center the map around the
        # Electronic City / Bommasandra area.

        traffic_map = folium.Map(
            location=[
                12.8450,
                77.6600
            ],
            zoom_start=14
        )


        # ----------------------------------------------------
        # Draw every road segment.
        # ----------------------------------------------------

        for i in range(len(speeds)):


            row = df[
                df["edge_id"] == (i + 1)
            ]


            if row.empty:

                continue


            row = row.iloc[0]


            geometry = str(
                row["geometry"]
            )


            coordinates = []


            # Convert LINESTRING geometry
            # into Folium's [latitude, longitude] format.

            try:

                geometry_string = (
                    geometry
                    .replace(
                        "LINESTRING (",
                        ""
                    )
                    .replace(
                        ")",
                        ""
                    )
                )


                for pair in geometry_string.split(","):

                    lon, lat = (
                        pair.strip().split()
                    )

                    coordinates.append([
                        float(lat),
                        float(lon)
                    ])


            except Exception:

                continue


            speed = float(
                speeds[i]
            )


            # Traffic visualization:
            #
            # Green  = faster road
            # Orange = moderate
            # Red    = slower

            if speed >= 25:

                color = "green"

            elif speed >= 18:

                color = "orange"

            else:

                color = "red"


            folium.PolyLine(

                coordinates,

                color=color,

                weight=5,

                tooltip=(
                    f"Road R{i} | "
                    f"Predicted Speed: "
                    f"{speed:.2f} km/h"
                )

            ).add_to(
                traffic_map
            )


        # ----------------------------------------------------
        # Highlight the fastest predicted road.
        # ----------------------------------------------------

        best_index = int(
            np.argmax(speeds)
        )


        best_row = df[
            df["edge_id"]
            == (best_index + 1)
        ]


        if not best_row.empty:

            best_row = best_row.iloc[0]


            geometry = str(
                best_row["geometry"]
            )


            try:

                geometry_string = (
                    geometry
                    .replace(
                        "LINESTRING (",
                        ""
                    )
                    .replace(
                        ")",
                        ""
                    )
                )


                best_coordinates = []


                for pair in geometry_string.split(","):

                    lon, lat = (
                        pair.strip().split()
                    )

                    best_coordinates.append([
                        float(lat),
                        float(lon)
                    ])


                folium.PolyLine(

                    best_coordinates,

                    color="blue",

                    weight=7,

                    tooltip=(
                        f"BEST ROAD: R{best_index} | "
                        f"{speeds[best_index]:.2f} km/h"
                    )

                ).add_to(
                    traffic_map
                )


            except Exception:

                pass


        # ----------------------------------------------------
        # Save generated map.
        # ----------------------------------------------------

        output_path = os.path.join(

            BASE_DIR,

            "static",

            "maps",

            "traffic_full.html"

        )


        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )


        traffic_map.save(
            output_path
        )


        return jsonify({

            "map_url":
            "/static/maps/traffic_full.html"

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# YOLO VEHICLE DETECTION
# ============================================================

@app.route(
    "/api/yolo_detect",
    methods=["POST"]
)
def yolo_detect():

    try:

        if yolo_model is None:

            return jsonify({
                "error":
                "YOLO model is not loaded."
            }), 500


        file = request.files["image"]


        upload_directory = os.path.join(
            BASE_DIR,
            "static",
            "uploads"
        )


        os.makedirs(
            upload_directory,
            exist_ok=True
        )


        save_path = os.path.join(
            upload_directory,
            file.filename
        )


        file.save(
            save_path
        )


        # Run YOLO inference on the uploaded image.

        results = yolo_model(
            save_path
        )[0]


        vehicle_count = len(
            results.boxes
        )


        return jsonify({

            "vehicle_count":
                int(vehicle_count),

            "filename":
                file.filename

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# SIMPLE ROUTE MAP
# ============================================================

@app.route(
    "/api/route_map",
    methods=["POST"]
)
def route_map():

    try:

        data = request.json


        start = data["start"]

        end = data["end"]


        route_map = folium.Map(
            location=start,
            zoom_start=14
        )


        folium.Marker(
            start,
            popup="Start",
            icon=folium.Icon(
                color="green"
            )
        ).add_to(route_map)


        folium.Marker(
            end,
            popup="Destination",
            icon=folium.Icon(
                color="red"
            )
        ).add_to(route_map)


        folium.PolyLine(
            [start, end],
            color="blue"
        ).add_to(route_map)


        map_path = os.path.join(

            BASE_DIR,

            "static",

            "maps",

            "route_map.html"

        )


        os.makedirs(
            os.path.dirname(map_path),
            exist_ok=True
        )


        route_map.save(
            map_path
        )


        return jsonify({

            "map_url":
            "/static/maps/route_map.html"

        })


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
    )
