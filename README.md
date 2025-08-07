# Map AI - Feature Matching & Map Cutting Service

This is a containerized service that provides comprehensive map processing functionalities:
1. **Feature Matching** for schematic maps using KAZE features
2. **Georeferenced Map Cutting** from multiple sources (OSM, Dutch BGT, etc.)
3. **Map Classification** using ResNet-50 based neural networks
4. **Combined Workflows** that integrate cutting and matching

All services are optimized for technical workflows and GIS applications.

## Project Structure

```
feature_matching_api/
├── app.py              ← FastAPI orchestrator with all endpoints
├── matcher.py          ← Feature matching logic (KAZE, optimized for schematic maps)
├── cutter.py           ← Map cutting logic (OSM, BGT with geometric buffering)
├── classifier_model.py ← Map classification model definitions
├── classifier_config.py ← Configuration for classifier
├── requirements.txt    ← Python dependencies
├── Dockerfile         ← Docker configuration  
├── docker-compose.yml ← Docker Compose configuration
└── README.md          ← This file

```

## Key Features

### 🗺️ **Map Cutting Service**
- **Multiple Map Sources**: OpenStreetMap, Dutch BGT, BRT-A, TOP10, BAG, Luchtfoto
- **Geometric Buffering**: Configurable buffer distances around input geometries
- **Multi-Geometry Support**: Points, LineStrings, Polygons, MultiGeometries
- **Format Flexibility**: GeoJSON, WKT, coordinate lists
- **Georeferencing**: Automatic world file (.pgw) generation
- **Overlay Combinations**: BGT+BAG, BRT-A+BAG, etc.

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

### 🤖 **Map Classification Service**
- **ResNet-50 Based**: Deep learning classifier for map type identification
- **12 Map Categories**: BRT-A, TOP10, BAG, BGT variants, Luchtfoto, OSM, combinations
- **Batch Processing**: Support for multiple images
- **Confidence Scores**: Detailed prediction probabilities for all classes

### 🔄 **Combined Workflows**
- **Cutout-and-Match**: Automatic buffer optimization with feature matching
- **URL-based Processing**: Accept image URLs instead of file uploads

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start the service
docker-compose up -d

# Check health
curl http://localhost:8002/health

# Stop the service
docker-compose down
```

### Using Docker Directly

```bash
# Build and run
docker build -t map-ai .
docker run -p 8002:8000 map-ai
```

### Access Points

- **API Documentation**: http://localhost:8002/docs
- **Health Check**: http://localhost:8002/health
- **Root Endpoint**: http://localhost:8002/

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
- `destination_pgw` (File, optional): PGW file for destination image georeferencing
- `overlay_transparency` (float, default: 0.6): Overlay transparency (0.0-1.0)
- `max_image_size` (int, default: 1500): Maximum image size for processing
- `output_format` (string, default: "json"): Output format ("json" or "files")

**Example:**
```bash
curl -X POST "http://localhost:8002/match-maps" \
  -F "source_image=@source.png" \
  -F "destination_image=@destination.png" \
  -F "destination_pgw=@destination.pgw" \
  -F "overlay_transparency=0.6" \
  -F "output_format=json"
```

#### POST `/match-maps-with-size-reduction` ⚡ **NEW**

Enhanced feature matching with size reduction optimization for faster processing.

This endpoint performs feature matching on reduced-size images for speed, then applies the transformation to the full-resolution source image. This approach significantly reduces processing time while maintaining the same final quality.

**Key Benefits:**
- ⚡ **Faster Processing**: Up to 4x faster feature detection and matching
- 💾 **Reduced Memory Usage**: Lower memory footprint during matching
- 🎯 **Same Quality**: Automatic homography scaling maintains accuracy
- 📏 **Flexible Sizing**: Configurable reduction level

**Parameters:**
- `source_image` (File, required): Source image to be warped (full resolution)
- `destination_image` (File, required): Destination/reference image
- `destination_pgw` (File, optional): PGW file for destination image georeferencing
- `overlay_transparency` (float, default: 0.6): Overlay transparency (0.0-1.0)
- `matching_image_size` (int, default: 800): Maximum dimension for matching phase

**Example:**
```bash
curl -X POST "http://localhost:8005/match-maps-with-size-reduction" \
  -F "source_image=@large_source.png" \
  -F "destination_image=@destination.png" \
  -F "destination_pgw=@destination.pgw" \
  -F "overlay_transparency=0.6" \
  -F "matching_image_size=800"
```

**Response includes timing and matching information:**
```json
{
  "success": true,
  "processing_time_seconds": 12.3,
  "matching_image_size": 800,
  "matches_count": 156,
  "inlier_ratio": 0.847,
  "files": {
    "overlay_result": "/download/session_123/warped_overlay_result.png",
    "warped_source": "/download/session_123/warped_source.png",
    "feature_matches": "/download/session_123/feature_matches.png"
  }
}
```

#### POST `/quick-cutting-matching-georeferencing` ⚡ **ULTRA-FAST**

Ultra-lightweight endpoint for quick cutting, matching, and georeferencing.

This endpoint is optimized for **maximum speed and minimal memory usage**:
- 🚀 **Ultra-Fast Processing**: Skips all overlay creation and unnecessary operations
- 💾 **Minimal Memory**: Uses lightweight warping without overlay blending  
- 🎯 **GeoTIFF Only**: Returns only the final georeferenced GeoTIFF download URL
- 🔄 **Auto-Optimization**: Automatically finds optimal buffer size for best matches
- ⚡ **Perfect for Production**: When you only need the final georeferenced result

**Key Benefits:**
- **Memory Efficient**: ~90% less memory usage compared to full workflow
- **Speed Optimized**: ~70% faster processing time
- **Clean Output**: Only essential GeoTIFF file, no overlays or visualizations
- **Smart Buffering**: Tests multiple buffer sizes to find the best feature matches

**Parameters:**
- `source_image` (File, required): Source image to be warped and georeferenced
- `geometry` (string, required): Geometry as GeoJSON, WKT, or coordinate list JSON string
- `map_type` (string, required): Map type from supported list
- `matching_image_size` (int, default: 800): Maximum dimension for matching phase

**Example:**
```bash
curl -X POST "http://localhost:8005/quick-cutting-matching-georeferencing" \
  -F "source_image=@source_cad.png" \
  -F "geometry=POINT(136000 455000)" \
  -F "map_type=bgt-omtrek" \
  -F "matching_image_size=800"
```

**Ultra-Minimal Response (only what you need):**
```json
{
  "success": true,
  "geotiff_url": "/download/session_123/warped_source.tif",
  "processing_time_seconds": 8.2,
  "best_buffer_meters": 800,
  "inlier_ratio": 0.847,
  "session_id": "session_20250110_142301_123456"
}
```

**Use Cases:**
- Production workflows requiring only final georeferenced output
- Batch processing of multiple CAD drawings
- Integration with GIS systems that only need GeoTIFF files
- Memory-constrained environments
- High-throughput processing scenarios

### 2. Map Cutting Endpoints

#### POST `/cut-out-georeferenced-map`

Cut out a georeferenced map section from various map sources based on geometry input.

Supports various geometry types (GeoJSON, WKT, coordinate lists) and multiple map sources including OpenStreetMap, Dutch BGT, BRT-A, aerial photography, and building layers. The cutter automatically calculates the bounding box of input geometries and adds a configurable geometric buffer around them.

**Supported Map Types:**
- **Basic maps**: osm, bgt-achtergrond, bgt-omtrek, bgt-standaard
- **Aerial photography**: luchtfoto (Dutch high-resolution)
- **Topographic maps**: brta, brta-omtrek, top10
- **Buildings**: bag (BAG buildings layer)
- **Overlay combinations**: bgt-bg-bag, bgt-bg-omtrek, brta-bag, brta-omtrek

**Parameters:**
- `geometry` (string, required): Geometry as GeoJSON, WKT, or coordinate list JSON string
- `map_type` (string, default: "osm"): Map type from supported list
- `buffer` (float, default: 800): Buffer distance in meters around geometry
- `output_format` (string, default: "json"): Output format ("json" or "files")

**Example:**
```bash
curl -X POST "http://localhost:8002/cut-out-georeferenced-map" \
  -F "geometry=POINT(136000 455000)" \
  -F "map_type=bgt-omtrek" \
  -F "buffer=500" \
  -F "output_format=json"
```

### 3. Combined Workflow Endpoints

#### POST `/cutout-and-match`

Cut out a map section based on geometry and perform feature matching with automatic buffer optimization.

This endpoint combines map cutting and feature matching with intelligent buffer selection:
1. Automatically tests multiple buffer sizes (100m, 500m, 800m, 2000m) around the input geometry
2. Cuts out georeferenced map sections for each buffer size
3. Performs KAZE feature matching for each buffer size
4. Selects the buffer size that produces the most inlier matches
5. Returns the best matching results with detailed comparison data

**Parameters:**
- `source_image` (File, required): Source image to be warped and matched
- `geometry` (string, required): Geometry as GeoJSON, WKT, or coordinate list JSON string
- `map_type` (string, required): Map type from supported list
- `overlay_transparency` (float, default: 0.6): Overlay transparency (0.0-1.0)
- `output_format` (string, default: "json"): Output format ("json" or "files")

**Example:**
```bash
curl -X POST "http://localhost:8002/cutout-and-match" \
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
2. Automatically tests multiple buffer sizes (100m, 500m, 800m, 2000m) around the input geometry
3. Cuts out georeferenced map sections for each buffer size
4. Performs KAZE feature matching for each buffer size
5. Selects the buffer size that produces the most inlier matches
6. Returns the best matching results with detailed comparison data

**Parameters:**
- `image_url` (string, required): URL of the source image to be warped and matched
- `geometry` (string, required): Geometry as GeoJSON, WKT, or coordinate list JSON string
- `map_type` (string, required): Map type from supported list
- `overlay_transparency` (float, default: 0.6): Overlay transparency (0.0-1.0)
- `output_format` (string, default: "json"): Output format ("json" or "files")

**Example:**
```bash
curl -X POST "http://localhost:8002/cutout-and-match-with-url" \
  -F "image_url=https://example.com/image.png" \
  -F "geometry=POINT(136000 455000)" \
  -F "map_type=osm" \
  -F "overlay_transparency=0.7" \
  -F "output_format=json"
```

**Note:** When `output_format` is set to "files", the endpoint returns the GeoTIFF file of the warped source image instead of the overlay result.

### 4. Map Classification Endpoints

#### POST `/classify-map`

Classify a map image into one of the trained map types.

This endpoint uses a ResNet-50 based classifier trained on map images.
Returns the predicted map type with confidence scores for all classes.

**Parameters:**
- `image` (File, required): Map image to classify (PNG, JPG, TIFF, BMP)
- `output_format` (string, default: "json"): Output format ("json" only supported)

**Example:**
```bash
curl -X POST "http://localhost:8002/classify-map" \
  -F "image=@map_sample.png" \
  -F "output_format=json"
```

#### POST `/classify`

Classify a single image (simplified endpoint).

**Parameters:**
- `file` (File, required): Image file to classify

**Example:**
```bash
curl -X POST "http://localhost:8002/classify" \
  -F "file=@map_sample.png"
```

#### POST `/classify-batch`

Classify multiple images in a single request.

**Parameters:**
- `files` (List[File], required): List of image files (max 10)

**Example:**
```bash
curl -X POST "http://localhost:8002/classify-batch" \
  -F "files=@image1.png" \
  -F "files=@image2.png" \
  -F "files=@image3.png"
```

#### GET `/classifier-info`

Get information about the classifier model and available classes.

Returns detailed information about the model architecture, available classes, preprocessing requirements, and usage instructions.

### 5. Utility Endpoints

#### GET `/`

Health check endpoint with service information.

Returns service status, version, timestamp, and available endpoints.

#### GET `/health`

Detailed health check endpoint.

Returns comprehensive health status for all service components (matcher, cutter, classifier).

#### GET `/download/{session_id}/{filename}`

Download processed files by session ID and filename.

**Parameters:**
- `session_id` (string): Session identifier
- `filename` (string): Filename to download

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
| `brta` | BRT-A standard visualization | Topographic colors | Topographic mapping |
| `brta-omtrek` | BRT-A grey visualization | Grey styling | Subdued topographic base |
| `top10` | BRT-TOP10 NL | Topographic colors | Dutch topographic standard |
| `bag` | BAG buildings layer | Transparent | Building footprints only |
| `bgt-bg-bag` | BGT background + BAG buildings | Combined | Buildings on BGT base |
| `bgt-bg-omtrek` | BGT background + outline | Combined | Outlines on BGT base |
| `brta-bag` | BRT-A + BAG buildings | Combined | Buildings on topographic |
| `brta-omtrek` | BRT-A + BGT outline | Combined | Mixed topographic/BGT |

## Classification Categories

The classifier recognizes 12 map types:
1. **BRT_A_standaard** - BRT-A standard topographic maps
2. **BRT_TOP10_NL** - TOP10NL topographic maps
3. **BAG_panden** - Building footprint maps
4. **BGT_achtergrond** - BGT background visualization
5. **BGT_standaard** - BGT standard visualization
6. **BGT_omtrek** - BGT outline visualization
7. **Luchtfoto** - Aerial photography
8. **Open_streetmap** - OpenStreetMap tiles
9. **Combinatie_BGT_achtergrond_BAG_panden** - BGT + BAG combination
10. **Combinatie_BGT_achtergrond_omtrekgerichte_visualisatie** - BGT + outline combination
11. **Combinatie_BRT_standaard_BAG_panden** - BRT-A + BAG combination
12. **Combinatie_BRT_standaard_omtrekgerichte_visualisatie** - BRT-A + outline combination

## Using the Modules Directly

### Map Cutting

```python
from cutter import cut_map

# Cut map with point geometry
result = cut_map(
    geometry_input=(136000, 455000),  # Utrecht center
    map_type="bgt-omtrek",
    output_dir="output"
)

# Cut map with complex geometry
result = cut_map(
    geometry_input={
        "type": "LineString", 
        "coordinates": [[136000, 455000], [136500, 455500]]
    },
    map_type="osm",
    output_dir="output"
)

if result.success:
    print(f"Map cutting successful!")
    print(f"Image size: {result.image.shape}")
    print(f"Bounds: {result.bounds}")
```

### Feature Matching

```python
from matcher import match_schematic_maps

result = match_schematic_maps(
    source_image_path="source.png",
    destination_image_path="destination.png", 
    output_dir="output",
    overlay_transparency=0.6
)

if result.success:
    print(f"Matches: {result.matches_count}")
    print(f"Quality: {result.inlier_ratio:.1%}")
```

## Technical Details

### Geometric Buffering

The service uses **true geometric buffering** via Shapely:
- Creates actual 500m buffer around geometry shape
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
- **Output**: Georeferenced with world files (.pgw)
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
- **GPU acceleration**: Automatic CUDA detection and usage

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

## Development

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python app.py

# Test individual modules
python cutter.py 136000 455000 500  # Test map cutting
python matcher.py source.png dest.png  # Test feature matching
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
- **rasterio**: GeoTIFF support (optional)
- **huggingface_hub**: Model downloading for GIM+Roma

## Migration Notes

This service evolved from the original KAZE feature matching code:

### What's New
- ✅ **Map cutting service** with multiple sources
- ✅ **BGT integration** for Dutch users
- ✅ **Geometric buffering** for accurate buffer zones
- ✅ **REST API** for easy integration
- ✅ **Docker deployment** for consistency
- ✅ **Fixed parameters** for simplified usage

### What's Preserved
- ✅ **KAZE feature matching** quality and settings
- ✅ **Georeferencing** capabilities
- ✅ **Quality assessment** algorithms
- ✅ **Multiple image format** support

The core functionality is enhanced with modern deployment practices and expanded capabilities for geographic workflows.

## Training Data Generation

### Generate Map Training Data

Use the `generate_map_training_data.py` script to create datasets for machine learning or analysis:

```bash
# Generate training data for all map types
python generate_map_training_data.py
```

**What it does:**
- Generates **200 random samples** for each of the 13 map types
- Uses **4 different buffer sizes**: 50m, 100m, 500m, 800m (50 samples each)
- Creates **separate folders** for each map type
- Places samples throughout the **Netherlands** (avoiding major water bodies)
- **Ensures all locations are within 50m of roads** (using OpenStreetMap data)
- Uses **proper coordinate conversion** (WGS84 → RD New)
- Includes **logging and progress tracking**

**Output structure:**
```
training_data/
├── osm/                    # 200 OpenStreetMap samples
├── bgt-achtergrond/        # 200 BGT background samples
├── bgt-omtrek/             # 200 BGT outline samples
├── bgt-standaard/          # 200 BGT standard samples
├── luchtfoto/              # 200 aerial photo samples
├── brta/                   # 200 BRT-A topographic samples
├── brta-omtrek/            # 200 BRT-A outline samples
├── top10/                  # 200 TOP10NL samples
├── bag/                    # 200 BAG building samples
├── bgt-bg-bag/             # 200 BGT+BAG overlay samples
├── bgt-bg-omtrek/          # 200 BGT+outline overlay samples
├── brta-bag/               # 200 BRT-A+BAG overlay samples
└── brta-omtrek/            # 200 BRT-A+outline overlay samples
```

**Features:**
- **Road proximity guarantee**: All locations within 50m of roads for visible infrastructure
- **Respectful to APIs**: Rate-limited requests to tile servers and OpenStreetMap
- **Geographic diversity**: Random coordinates across Netherlands
- **Error handling**: Continues on failures, reports statistics
- **Descriptive filenames**: Include map type, buffer size, sample number
- **Images only**: PNG files without world files for clean datasets
- **Comprehensive logging**: Progress and error tracking

**Configuration:**
You can modify the script parameters:
```python
SAMPLES_PER_MAP_TYPE = 200        # Total samples per map type
BUFFER_SIZES = [50, 100, 500, 800]  # Buffer sizes in meters
TARGET_WIDTH = 2048               # Output image width
```

**Usage notes:**
- Script takes **several hours** to complete (2600+ total samples)
- **Slower due to road checking**: Uses OpenStreetMap API for location validation
- Can be **interrupted** safely with Ctrl+C
- **Excluded from Docker**: Use locally for dataset creation
- Generates **training_data_generation.log** for monitoring
- **Requires internet**: Needs access to OpenStreetMap Overpass API

This is useful for creating datasets for:
- Map classification models
- Feature detection training
- Visual analysis of different map types
- Geographic data science projects

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