# object-detection-monitoring

## Features

### **Runtime Monitoring anomaly Detection**
- **Lost Object Detection**: Identifies when tracked objects disappear from the scene
- **Label Switching Detection**: Detects when object classifications change unexpectedly
- **Sudden Change Detection**: Identifies abrupt movements or appearance changes
- **Wrong Location Detection**: Validates object positions against prior knowledge
- **Unusual Size Detection**: Detects objects with abnormal size characteristics

## Requirements
- `python>=3.10`
- `uv` : Install uv as python project manager from [here](https://github.com/astral-sh/uv.git)


## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/Rayzheng227/object-detection-monitoring.git
cd object-detection-monitoring
```

2. **Sync virtual environment by uv** (recommended)
```bash
uv sync
```
or  **Install dependencies by pip**:
```bash
pip install -r requirements.txt
```


##  Start

### Basic Usage
```bash
# Run basic tracking (geometric features only)
uv run run_basic.py

# Run advanced tracking with anomaly detection
uv run run.py
```


## 📁 Project Structure

```
Tracking/
├── README.md
├── requirements.txt
├── pyproject.toml                
├── uv.lock                      
├── __init__.py                  
├── run.py                       # Main tracking script
├── run_basic.py                # Basic tracking script
├── src/                        # Source code modules
│   ├── monitors/               # Monitoring and anomaly detection
│   │   ├── vmonitor.py         
│   │   └── anomaly_vmonitor.py 
│   ├── extractors/             # Feature extraction modules
│   │   ├── extractor.py        
│   │   └── shuffenet.py        
│   └── tracker/                # Core tracking algorithms
│       ├── kcftracker_mod.py   
│       ├── fhog.py            
│       ├── convert.py         
├── data/                       # Dataset and input files
│   ├── detections/            # Object detection results
│   ├── labels/                # Ground truth labels
│   └── videos/                # Video data
├── sem/                       # Semantic analysis results
```

