# 🚦 Traffic Prediction & Route Optimization

A **spatio-temporal traffic prediction and route recommendation system** that combines **Graph Attention Networks (GAT)** and **Long Short-Term Memory (LSTM)** to predict traffic speeds across connected road segments.

The system learns:

* 🗺️ **Spatial dependencies** between connected roads using GAT
* ⏱️ **Temporal dependencies** from historical traffic patterns using LSTM
* 🛣️ **Prediction-driven route recommendations** based on predicted road speeds

---

## 📌 Project Overview

Traffic conditions depend on both the **road network structure** and how traffic changes over time.

A conventional LSTM can learn temporal patterns, but it does not explicitly understand relationships between connected roads. Similarly, a graph-based model can capture spatial relationships but may not effectively model long-term temporal patterns.

This project combines **Graph Attention Networks + LSTM** to model both dimensions.

The system takes a selected:

* Date
* Time
* Traffic scenario

and generates predicted traffic speeds for **30 selected road segments**.

These predictions are then used to provide a simple route recommendation and visualize traffic conditions across the road network.

---

# 🧠 System Architecture

```text
                 User Input
              Date / Time / Scenario
                       │
                       ▼
              ┌─────────────────┐
              │  Preprocessing  │
              │  preprocess.py  │
              └────────┬────────┘
                       │
                  12 × 30 × 8
                       │
                       ▼
              ┌─────────────────┐
              │  Road Network   │
              │     Graph       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │       GAT       │
              │ Spatial Learning│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │      LSTM       │
              │Temporal Learning│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     Fusion      │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Predicted Speeds│
              │ 30 Road Segments│
              └────────┬────────┘
                       │
              ┌────────┴─────────┐
              ▼                  ▼
       Prediction UI       Traffic Map
```

---

# 🗺️ Road Network Representation

The road network is represented as a **graph**.

* Each selected road segment is represented as a **node**.
* Road segments sharing physical endpoints are connected.
* Graph connectivity is represented using PyTorch Geometric's `edge_index`.
* The current system operates on **30 selected road segments**.

Road-network information is stored in:

```text
data/ecity_edges.csv
```

---

# 📊 Model Input

The model receives a historical traffic sequence with the shape:

```text
12 × 30 × 8
```

| Dimension | Meaning                   |
| --------- | ------------------------- |
| 12        | Historical time steps     |
| 30        | Road segments             |
| 8         | Features per road segment |

The preprocessing layer converts the selected timestamp and traffic scenario into the model input representation.

### Supported Scenarios

* Normal
* Rain
* Event

---

# 🧠 Hybrid GAT-LSTM Model

The model is implemented in:

```text
utils/hybrid_model.py
```

## Graph Attention Network

The **GAT** component captures spatial dependencies between connected road segments.

Instead of treating all neighboring roads equally, the attention mechanism learns different importance weights for neighboring nodes.

### Configuration

* GAT hidden dimension: **64**
* Attention heads: **4**

---

## LSTM

The **LSTM** component captures temporal dependencies across historical traffic observations.

### Configuration

* LSTM hidden dimension: **128**

---

## Feature Fusion

The spatial and temporal representations are combined before the final prediction layer.

```text
GAT Representation
        +
LSTM Representation
        │
        ▼
      Fusion
        │
        ▼
Traffic Speed Prediction
```

The model produces **one predicted traffic speed for each of the 30 road segments**.

---

# 💾 Trained Model

The trained model checkpoint is stored at:

```text
models/hybrid_model_final.pt
```

During deployment, the checkpoint is loaded into the `HybridGAT_LSTM` architecture.

The application performs **inference using the trained model** rather than retraining the model for every prediction request.

---

# ⚙️ Backend

The backend is implemented using **Flask**.

Main backend file:

```text
app.py
```

## Traffic Prediction

### `POST /api/hybrid_predict`

Receives:

* Date
* Time
* Scenario

and returns predicted traffic speeds for the road segments.

---

## Route Recommendation

### `POST /api/route`

Uses the predicted traffic speeds to identify the fastest predicted road within the selected road-index range.

The current implementation recommends the road segment with the **highest predicted speed**.

---

## Traffic Map

### `POST /api/route_map_full`

Generates a **Folium-based visualization** of predicted traffic conditions across the road network.

---

# 🖥️ Frontend

The frontend is built using:

* HTML
* CSS
* JavaScript

### Templates

```text
templates/
├── index.html
├── predict.html
└── map.html
```

### Home Page

Provides an overview of the traffic prediction system and navigation to the prediction and map interfaces.

### Prediction Page

Allows users to select:

* Date
* Time
* Traffic scenario

and request predictions from the trained Hybrid GAT-LSTM model.

The predicted speeds for the 30 road segments are displayed in the interface.

### Map Page

Displays the generated traffic visualization using **Folium**.

---

# 🔄 Prediction Pipeline

When a user requests a prediction:

```text
1. User selects date, time and scenario
                    │
                    ▼
2. Flask receives the request
                    │
                    ▼
3. preprocess.py generates traffic sequence
                    │
                    ▼
4. Input is converted to a PyTorch tensor
                    │
                    ▼
5. Batch dimension is added
                    │
                    ▼
6. Road graph is supplied through edge_index
                    │
                    ▼
7. Hybrid GAT-LSTM performs inference
                    │
                    ▼
8. Predicted traffic speeds are generated
                    │
                    ▼
9. Flask returns predictions as JSON
                    │
                    ▼
10. Frontend displays the results
```

---

# 🛣️ Route Recommendation

After obtaining predicted traffic speeds, the application compares the predicted speeds across a selected road-index range.

The road segment with the **highest predicted speed** is returned as the recommended road.

This provides a simple prediction-driven route selection mechanism.

> **Note:** The current route recommendation is based on predicted road speed rather than a full multi-road shortest-path optimization algorithm. A future version can incorporate graph-based path optimization, travel time, distance, congestion penalties, and turn costs.

---

# 📁 Project Structure

```text
traffic-prediction-route-optimization/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── ecity_edges.csv
│
├── models/
│   └── hybrid_model_final.pt
│
├── utils/
│   ├── graph_utils.py
│   ├── hybrid_model.py
│   └── preprocess.py
│
└── templates/
    ├── index.html
    ├── predict.html
    └── map.html
```

---

# 🛠️ Technologies Used

### Machine Learning

* Python
* PyTorch
* PyTorch Geometric
* NumPy
* Pandas

### Deep Learning

* Graph Attention Networks (GAT)
* Long Short-Term Memory (LSTM)
* Hybrid GAT-LSTM Architecture

### Backend

* Flask
* Gunicorn

### Visualization

* Folium
* HTML
* CSS
* JavaScript

### Deployment

* Render

---

# 🚀 Running Locally

## 1. Clone the Repository

```bash
git clone https://github.com/Mounanjali19/traffic-prediction-route-optimization.git
cd traffic-prediction-route-optimization
```

## 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the Application

```bash
python app.py
```

The application will be available at:

```text
http://localhost:5000
```

---

# ☁️ Deployment

The application can be deployed as a Flask web service.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn --timeout 120 app:app
```

---

# ⚠️ Current Data Limitation

The current deployed version **does not use a live traffic API or real-time traffic sensor feed**.

Instead, the preprocessing layer constructs the traffic feature sequence based on the selected:

* Timestamp
* Traffic scenario

The generated sequence is then passed to the trained **Hybrid GAT-LSTM model** for prediction.

This means the current system demonstrates the **prediction and deployment pipeline**, but it should not be interpreted as a live traffic forecasting system.

---

# 🔮 Future Improvements

Several extensions can improve the system:

### Real-Time Traffic

* Integrate live traffic APIs
* Integrate real-time traffic sensor streams
* Continuously update traffic predictions

### Larger Road Networks

* Expand beyond the current 30 road segments
* Support larger metropolitan road networks
* Incorporate more detailed road connectivity

### Advanced Route Optimization

* Replace simple highest-speed selection with graph-based path optimization
* Consider travel time and road distance
* Incorporate congestion penalties
* Consider turn costs and route constraints

### Model Improvements

* Add more historical traffic data
* Incorporate weather information
* Include events and special-day features
* Compare against additional baseline models
* Perform systematic model benchmarking and ablation studies
* Automate model retraining with newly collected traffic data

---

# 🎯 Project Goals

The project demonstrates how **graph-based deep learning and temporal sequence modeling** can be combined to model traffic behavior across a connected road network.

The overall objective is to build a foundation for a scalable traffic intelligence system capable of:

```text
Traffic Data
     │
     ▼
Spatio-Temporal Modeling
     │
     ▼
Traffic Speed Prediction
     │
     ▼
Route Optimization
     │
     ▼
Traffic Visualization
```

---

## 👩‍💻 Author

**Mounanjali L**

GitHub:
https://github.com/Mounanjali19

---

## ⭐ If You Found This Project Interesting

Consider giving the repository a ⭐ on GitHub!
