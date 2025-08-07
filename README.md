# Map AI - Feature Matching & Map Cutting Service

This is a containerized service that provides comprehensive map processing functionalities using a modern, refactored architecture:

1. **Feature Matching** for schematic maps using KAZE features and aerial imagery using GIM+Roma models
2. **Georeferenced Map Cutting** from multiple sources (OSM, Dutch BGT, BRT-A, Luchtfoto, etc.)
3. **Combined Workflows** that integrate cutting and matching with automatic buffer optimization
4. **Session Management** for tracking and organizing processing results

All services are optimized for technical workflows, GIS applications, and production environments.

## Project Architecture

The application has been refactored into a clean, modular architecture:

```
map_feature_matching_model/
├── app_refactored.py              ← FastAPI orchestrator with all endpoints
├── services/                      ← Core business logic services
│   ├── feature_matching_service.py    ← Feature matching logic (KAZE + GIM+Roma)
│   ├── map_cutting_service.py         ← Map cutting and tile downloading
│   ├── image_processing_service.py    ← Image manipulation and optimization
│   ├── tile_service.py                ← Tile downloading and stitching
│   ├── coordinate_transformation_service.py ← Coordinate system transformations
│   ├── georeferencing_service.py      ← Georeferencing and world file handling
│   ├── visualization_service.py       ← Result visualization and overlays
│   ├── matching_service.py            ← Core matching algorithms
│   └── response_service.py            ← Response formatting and file management
├── utils/                         ← Shared utilities
│   ├── coordinate_utils.py            ← Coordinate calculations and transformations
│   ├── geometry_utils.py              ← Geometric operations and buffering
│   ├── file_utils.py                  ← File I/O and management
│   ├── validation_utils.py            ← Input validation and sanitization
│   ├── session_utils.py               ← Session management
│   ├── response_utils.py              ← Response formatting helpers
│   ├── worldfile_utils.py             ← World file (.pgw) operations
│   ├── tile_server_utils.py           ← Tile server configurations
│   └── memory_utils.py                ← Memory optimization utilities
├── models/                        ← Data models and schemas
│   ├── request_models.py              ← Request data models
│   └── response_models.py             ← Response data models
├── third_party/                   ← Third-party components
│   ├── roma_minimal.py                ← GIM+Roma matcher implementation
│   ├── setup_roma.py                  ← ROMA model setup script
│   └── RoMa/                          ← RoMa repository (auto-downloaded)
├── requirements.txt               ← Python dependencies
├── Dockerfile                     ← Docker configuration  
├── docker-compose.yml             ← Docker Compose configuration
└── README.md                      ← This file
```

## Key Features

### 🗺️ **Map Cutting Service**
- **Multiple Map Sources**: OpenStreetMap, Dutch BGT, BRT-A, TOP10, BAG, Luchtfoto
- **Geometric Buffering**: Configurable buffer distances around input geometries
- **Multi-Geometry Support**: Points, LineStrings, Polygons, MultiGeometries
- **Format Flexibility**: GeoJSON, WKT, coordinate lists
- **Georeferencing**: Automatic world file (.pgw) generation
- **Overlay Combinations**: BGT+BAG, BRT-A+BAG, etc.
- **Smart Zoom Selection**: Automatic optimization for different map types

### 🔍 **Feature Matching Service** 
- **Dual Matching Engines**: 
  - **KAZE features** for schematic maps (optimized for technical diagrams)
  - **GIM+Roma model** for aerial imagery (luchtfoto) matching
- **Intelligent Matcher Selection**: Automatically selects the best matcher based on map type
- **Multiple Fallbacks**: KAZE → ORB → SIFT detectors for schematic maps
- **Size Reduction Optimization**: Fast matching on reduced images with full-resolution warping
- **Quality Assessment**: Automatic matching quality evaluation
- **Georeferencing**: Transfers coordinate systems via world files
- **Buffer Optimization**: Automatically tests multiple buffer sizes

### 🔄 **Combined Workflows**
- **Cutout-and-Match**: Automatic buffer optimization with feature matching
- **URL-based Processing**: Accept image URLs instead of file uploads
- **Session Management**: Organized file handling with session-based storage
- **Smart Caching**: Persistent model checkpoints to avoid re-downloading

### ⚡ **Performance & Reliability**
- **Memory Optimization**: Efficient memory usage with automatic cleanup
- **CPU/GPU Support**: Automatic device detection for optimal performance
- **Persistent Caching**: Model checkpoints cached between container restarts
- **Error Handling**: Comprehensive error handling and logging
- **Resource Management**: Configurable memory limits and optimization

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start the service
docker-compose up -d

# Check health
curl http://localhost:8005/health

# Stop the service
docker-compose down
```

### Using Docker Directly

```bash
# Build and run
docker build -t map-ai .
docker run -p 8005:8000 map-ai
```

### Access Points

- **API Documentation**: http://localhost:8005/docs
- **Health Check**: http://localhost:8005/health
- **Root Endpoint**: http://localhost:8005/

## API Reference

### 1. Feature Matching Endpoints

#### POST `/match-maps`

Perform feature matching between two images with optional georeferencing.

This endpoint automatically selects the appropriate matching algorithm:
- **KAZE features** for schematic maps and technical diagrams
- **GIM+Roma model** for aerial imagery (luchtfoto) 

If a PGW file is provided for the destination image, the output will be georeferenced.

**Parameters:**
- `source_image` (File, required): Source image to be warped (PNG, JPG, TIFF, BMP)
- `destination_image` (File, required): Destination/reference image (PNG, JPG, TIFF, BMP)
- `destination_pgw` (File, required): PGW file for destination image georeferencing
- `overlay_transparency` (float, default: 0.6): Overlay transparency (0.0-1.0)
- `output_format` (string, default: "json"): Output format ("json" or "files")
- `traffic_decree_id` (string, optional): Optional traffic decree ID for session naming

**Example:**
```bash
curl -X POST "http://localhost:8005/match-maps" \
  -F "source_image=@source.png" \
  -F "destination_image=@destination.png" \
  -F "destination_pgw=@destination.pgw" \
  -F "overlay_transparency=0.6" \
  -F "output_format=json"
```

### 2. Map Cutting Endpoints

#### POST `/cut-out-georeferenced-map`

Cut out a georeferenced map section from various map sources based on geometry input.

Supports various geometry types (GeoJSON, WKT, coordinate lists) and multiple map sources including OpenStreetMap, Dutch BGT, BRT-A, aerial photography, and building layers. The cutter automatically calculates the bounding box of input geometries and adds a configurable geometric buffer around them.

**Supported Map Types:**
- **Basic maps**: osm, bgt-achtergrond, bgt-omtrek, bgt-standaard
- **Aerial photography**: luchtfoto, luchtfoto-2022 (Dutch high-resolution)
- **Topographic maps**: brta, brta-omtrek, top10
- **Buildings**: bag (BAG buildings layer)
- **Overlay combinations**: bgt-bg-bag, bgt-bg-omtrek, brta-bag, brta-omtrek

**Parameters:**
- `geometry` (string, required): Geometry as GeoJSON, WKT, or coordinate list JSON string
- `map_type` (string, default: "osm"): Map type from supported list
- `buffer` (float, default: 800): Buffer distance in meters around geometry
- `output_format` (string, default: "json"): Output format ("json" or "files")
- `traffic_decree_id` (string, optional): Optional traffic decree ID for session naming

**Example:**
```bash
curl -X POST "http://localhost:8005/cut-out-georeferenced-map" \
  -F "geometry=POINT(136000 455000)" \
  -F "map_type=bgt-omtrek" \
  -F "buffer=500" \
  -F "output_format=json"
```

### 3. Combined Workflow Endpoints

#### POST `/cutout-and-match`

Cut out a map section based on geometry and perform feature matching with automatic buffer optimization.

This endpoint combines map cutting and feature matching with intelligent buffer selection:
1. Automatically tests multiple buffer sizes around the input geometry
2. Cuts out georeferenced map sections for each buffer size
3. Performs feature matching for each buffer size
4. Selects the buffer size that produces the most inlier matches
5. Returns the best matching results with detailed comparison data

**Parameters:**
- `source_image` (File, required): Source image to be warped and matched
- `geometry` (string, required): Geometry as GeoJSON, WKT, or coordinate list JSON string
- `map_type` (string, required): Map type from supported list
- `overlay_transparency` (float, default: 0.6): Overlay transparency (0.0-1.0)
- `output_format` (string, default: "json"): Output format ("json" or "files")
- `traffic_decree_id` (string, optional): Optional traffic decree ID for session naming

**Example:**
```bash
curl -X POST "http://localhost:8005/cutout-and-match" \
  -F "source_image=@source.png" \
  -F "geometry=POINT(136000 455000)" \
  -F "map_type=osm" \
  -F "overlay_transparency=0.7" \
  -F "output_format=json"
```

**Note:** When `output_format` is set to "files", the endpoint returns the GeoTIFF file of the warped source image instead of the overlay result.

#### POST `/cutout-and-match-with-url`

Cut out a map section based on geometry and perform feature matching with automatic buffer optimization using an image URL.

Same functionality as `/cutout-and-match` but accepts an image URL instead of file upload:
1. Downloads the source image from the provided URL
2. Automatically tests multiple buffer sizes around the input geometry
3. Cuts out georeferenced map sections for each buffer size
4. Performs feature matching for each buffer size
5. Selects the buffer size that produces the most inlier matches
6. Returns the best matching results with detailed comparison data

**Parameters:**
- `image_url` (string, required): URL of the source image to be warped and matched
- `geometry` (string, required): Geometry as GeoJSON, WKT, or coordinate list JSON string
- `map_type` (string, required): Map type from supported list
- `overlay_transparency` (float, default: 0.6): Overlay transparency (0.0-1.0)
- `output_format` (string, default: "json"): Output format ("json" or "files")
- `traffic_decree_id` (string, optional): Optional traffic decree ID for session naming

**Example:**
```bash
curl -X POST "http://localhost:8005/cutout-and-match-with-url" \
  -F "image_url=https://example.com/image.png" \
  -F "geometry=POINT(136000 455000)" \
  -F "map_type=osm" \
  -F "overlay_transparency=0.7" \
  -F "output_format=json"
```

**Note:** When `output_format` is set to "files", the endpoint returns the GeoTIFF file of the warped source image instead of the overlay result.

### 4. Utility Endpoints

#### GET `/`

Health check endpoint with service information.

Returns service status, version, timestamp, and available endpoints.

#### GET `/health`

Detailed health check endpoint.

Returns comprehensive health status for all service components (matcher, cutter).

#### GET `/download/{session_id}/{file_path:path}`

Download processed files by session ID and file path (supports subdirectories).

**Parameters:**
- `session_id` (string): Session identifier
- `file_path` (string): File path to download (supports subdirectories)

**Example:**
```bash
curl "http://localhost:8005/download/session_123/warped_overlay_result.png" -o result.png
curl "http://localhost:8005/download/session_123/buffer_500m_cutout/feature_matches.png" -o matches.png
```

#### GET `/sessions`

List all active sessions.

Returns information about all active processing sessions including files and creation timestamps.

#### DELETE `/sessions/{session_id}`

Clean up a specific session's files.

**Parameters:**
- `session_id` (string): Session identifier to clean up

## Map Type Reference

| Map Type | Description | Background | Best For |
|----------|-------------|------------|----------|
| `osm` | OpenStreetMap standard tiles | Standard colors | General mapping |
| `bgt-achtergrond` | Dutch BGT background visualization | Filled areas | Detailed base maps |
| `bgt-omtrek` | Dutch BGT outline visualization | White background | Technical analysis, overlays |
| `bgt-standaard` | Dutch BGT standard visualization | Default BGT styling | Standard BGT usage |
| `luchtfoto` | Dutch aerial photography | Natural colors | High-resolution imagery |
| `luchtfoto-2022` | Winter Dutch aerial photography | Natural colors (winter) | High-resolution winter imagery |
| `brta` | BRT-A standard visualization | Topographic colors | Topographic mapping |
| `brta-omtrek` | BRT-A grey visualization | Grey styling | Subdued topographic base |
| `top10` | BRT-TOP10 NL | Topographic colors | Dutch topographic standard |
| `bag` | BAG buildings layer | Transparent | Building footprints only |
| `bgt-bg-bag` | BGT background + BAG buildings | Combined | Buildings on BGT base |
| `bgt-bg-omtrek` | BGT background + outline | Combined | Outlines on BGT base |
| `brta-bag` | BRT-A + BAG buildings | Combined | Buildings on topographic |
| `brta-omtrek` | BRT-A + BGT outline | Combined | Mixed topographic/BGT |

## Technical Details

### Service Architecture

The refactored architecture separates concerns into distinct services:

- **Feature Matching Service**: Handles KAZE and GIM+Roma matching algorithms
- **Map Cutting Service**: Orchestrates tile downloading and map creation
- **Tile Service**: Downloads and stitches tiles from various map sources
- **Coordinate Transformation Service**: Handles coordinate system conversions
- **Georeferencing Service**: Manages world file creation and GeoTIFF output
- **Image Processing Service**: Handles image manipulation and optimization
- **Visualization Service**: Creates overlays and result visualizations

### Geometric Buffering

The service uses **true geometric buffering** via Shapely:
- Creates actual buffer distances around geometry shapes
- More accurate than rectangular bounding box buffering
- Handles complex geometries properly (LineStrings, Polygons)
- Buffer distance is consistent around entire geometry

### BGT Optimizations

For Dutch BGT (Basisregistratie Grootschalige Topografie):
- **Minimum zoom level 18**: Prevents white/empty tiles
- **White backgrounds** for omtrek layers: Better visibility of line outlines
- **Larger image allowance**: Up to 5x target width to maintain detail
- **Proper transparency handling**: RGBA tiles composited correctly

### Coordinate Systems

- **Input**: RD New (EPSG:28992) coordinates expected
- **Output**: Georeferenced with world files (.pgw) and GeoTIFF
- **Internal**: Web Mercator (EPSG:3857) for tile fetching
- **Transformations**: Automatic via pyproj

### Feature Matching Configuration

#### Schematic Maps (KAZE-based)
Optimized for technical diagrams and CAD drawings:
- **KAZE detectors**: Extended 128-byte descriptors with nonlinear diffusion
- **Multi-detector fallback**: KAZE → ORB → SIFT for robustness
- **Edge enhancement**: Aggressive preprocessing for line-based content
- **Quality thresholds**: Automatic assessment with inlier ratio analysis

#### Aerial Imagery (GIM+Roma-based)
Optimized for satellite and aerial photography (luchtfoto):
- **GIM+Roma model**: State-of-the-art deep learning matcher
- **DINOv2 backbone**: Strong visual feature extraction
- **Global-to-local matching**: Coarse-to-fine correspondence estimation
- **Uncertainty estimation**: Robust matching with confidence scores
- **CPU/GPU acceleration**: Automatic device detection and usage
- **Persistent caching**: Model checkpoints cached between runs

## Quality Assessment

### Map Cutting
- **Tile success rate**: Percentage of successfully downloaded tiles
- **Zoom optimization**: Automatic zoom level selection
- **Coverage validation**: Ensures adequate map coverage

### Feature Matching
- **Excellent**: ≥30% inlier ratio
- **Good**: ≥20% inlier ratio
- **Fair**: ≥10% inlier ratio  
- **Poor**: <10% inlier ratio

## Environment Variables

- `PORT`: Service port (default: 8000)
- `UPLOAD_DIR`: Upload directory (default: "uploads")
- `OUTPUT_DIR`: Output directory (default: "outputs")
- `OMP_NUM_THREADS`: OpenMP thread limit (default: "1")
- `MKL_NUM_THREADS`: MKL thread limit (default: "1")

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python app_refactored.py

# Test individual services
python -c "from services.map_cutting_service import MapCuttingService; service = MapCuttingService(); print('Service initialized')"
```

### Dependencies

Key dependencies:
- **FastAPI + Uvicorn**: REST API framework
- **OpenCV**: Image processing and computer vision (KAZE, ORB, SIFT detectors)
- **PyTorch**: Deep learning framework for GIM+Roma model
- **Shapely**: Geometric operations and buffering
- **pyproj**: Coordinate system transformations
- **Pillow**: Image I/O and manipulation
- **requests**: HTTP tile downloading
- **rasterio**: GeoTIFF support
- **huggingface_hub**: Model downloading for GIM+Roma

## Model Attributions and Setup

### GIM+Roma Feature Matcher

This project includes the GIM+Roma model for aerial imagery matching, which combines:

- **GIM (Global Image Matcher)**: Learning Generalizable Image Matcher From Internet Videos
- **Roma**: Robust feature matching with uncertainty estimation  
- **DINOv2**: Strong visual backbone for feature extraction

#### Model Setup

The GIM+Roma model is automatically downloaded on first use. To manually set up:

```bash
python third_party/setup_roma.py
```

This will:
- Install required Python packages
- Clone the RoMa repository to `third_party/RoMa`
- Download the GIM+Roma model (`gim_roma_100h.ckpt`)
- Apply CPU compatibility patch for Mac systems
- Verify your PyTorch installation

**Persistent Model Caching:**

Models are cached in Docker volumes to avoid re-downloading:
- **Hugging Face cache**: `/root/.cache/huggingface` (persistent Docker volume)
- **Local checkpoints**: `./ROMA_checkpoints` (host directory mount)
- **First run**: Downloads ~1.2GB of model data
- **Subsequent runs**: Loads instantly from cache

**CPU Compatibility for Mac Systems:**

The setup script automatically applies a compatibility patch to ensure the model runs on CPU (essential for Mac systems without CUDA). This modifies the `half` precision default in `romatch/utils/kde.py`:

```python
# Original (GPU-optimized)
def kde(x, std = 0.1, half = True, down = None):

# Patched (CPU-compatible)  
def kde(x, std = 0.1, half = False, down = None):
```

**For GPU Systems:**
If you have a CUDA-compatible GPU and want maximum performance, you can skip this patch by manually setting `half = True` back in the `kde.py` file after setup. However, the CPU-compatible version will work on both CPU and GPU systems, just with slightly reduced performance on GPU.

#### Model Details

- **Model file**: `gim_roma_100h.ckpt` (automatically downloaded from Hugging Face)
- **Platform compatibility**: Runs on both CPU and GPU with automatic detection
- **Memory requirements**: 
  - **CPU**: 4GB+ RAM recommended
  - **GPU**: 4GB+ VRAM recommended for optimal performance
- **Input resolution**: Automatically handled with coarse-to-fine matching
- **Mac compatibility**: CPU-only execution ensured via automatic patching

#### Usage Context

The GIM+Roma model is specifically used for `luchtfoto` (aerial photography) map types because traditional KAZE feature matching was not sufficiently robust for this imagery type. For all other map types (schematic maps, technical diagrams), the system uses the optimized KAZE-based pipeline.

### Citations

If you use this software in academic work, please cite the relevant papers:

```bibtex
@article{edstedt2023roma,
  title={RoMa: A Lightweight Multi-Camera 3D Depth Estimation Framework},
  author={Edstedt, Johan and Athanasiadis, Ioannis and Wadenbäck, Mårten and Felsberg, Michael},
  journal={arXiv preprint arXiv:2305.15404},
  year={2023}
}

@article{shen2024gim,
  title={GIM: Learning Generalizable Image Matcher From Internet Videos},
  author={Shen, Xuelun and Hu, Zhipeng and Chen, Minhang and Luo, Zijun and Shen, Hao and others},
  journal={ICLR},
  year={2024}
}

@article{alcantarilla2012kaze,
  title={KAZE Features},
  author={Alcantarilla, Pablo F and Bartoli, Adrien and Davison, Andrew J},
  journal={European Conference on Computer Vision (ECCV)},
  year={2012}
}
```

### Model Sources

- **RoMa**: [GitHub - Parskatt/RoMa](https://github.com/Parskatt/RoMa)
- **GIM**: [GitHub - xuelunshen/gim](https://github.com/xuelunshen/gim)
- **DINOv2**: [GitHub - facebookresearch/dinov2](https://github.com/facebookresearch/dinov2)
- **KAZE**: [GitHub - pablofdezalc/kaze](https://github.com/pablofdezalc/kaze)

## License

This project is licensed under the **MIT License**.

### Third-Party Licenses

This project includes several third-party components with their respective licenses:

#### Feature Matching Models
- **ROMA matcher model**: MIT License
- **GIM model**: MIT License  
- **DINOv2 model weights and components**: [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- **KAZE features (via OpenCV)**: Apache License 2.0 (OpenCV 4.5.0+) or BSD-3-Clause (OpenCV 4.4.0 and lower)

#### Core Dependencies
- **OpenCV**: Apache License 2.0 (newer versions) or BSD-3-Clause (older versions)
- **PyTorch**: BSD-3-Clause License
- **FastAPI**: MIT License

All third-party licenses are compatible with commercial use and do not impose restrictions on derivative works beyond standard attribution requirements.

For complete license details, see the [NOTICE](./NOTICE) file.