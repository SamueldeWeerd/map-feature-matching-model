#!/usr/bin/env python3
"""
Refactored FastAPI Orchestrator for Map AI Processing

This is a demonstration of how the app.py can be refactored to separate
concerns and make the code more maintainable.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from datetime import datetime

# Import our services and utilities
from services.feature_matching_service import FeatureMatchingService
from services.map_cutting_service import MapCuttingService
from utils.file_utils import FileManager
from utils.validation_utils import ValidationUtils, validate_common_inputs
from utils.response_utils import ResponseFormatter
from utils.session_utils import SessionManager

# Set environment variables to limit thread usage
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Map AI Processing Service",
    description="A service for processing maps with feature matching, cutting, and classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory configuration
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

# Initialize services
file_manager = FileManager(UPLOAD_DIR, OUTPUT_DIR)
session_manager = SessionManager(UPLOAD_DIR, OUTPUT_DIR)
feature_matching_service = FeatureMatchingService()
map_cutting_service = MapCuttingService()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Map AI Processing Service",
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "available_endpoints": {
            "feature_matching": "/match-maps",
            "feature_matching_with_size_reduction": "/match-maps-with-size-reduction",
            "map_cutting_geometry": "/cut-out-georeferenced-map", 
            "cutout_and_match": "/cutout-and-match",
            "cutout_and_match_with_url": "/cutout-and-match-with-url",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "matcher": "available",
            "cutter": "available",
        }
    }


@app.post("/match-maps")
async def match_maps(
    source_image: UploadFile = File(..., description="Source image to be warped"),
    destination_image: UploadFile = File(..., description="Destination/reference image"),
    destination_pgw: UploadFile = File(..., description="Optional PGW file for destination image georeferencing"),
    overlay_transparency: float = Form(0.6, description="Overlay transparency (0.0-1.0)"),
    output_format: str = Form("json", description="Output format: 'json' or 'files'"),
    traffic_decree_id: str = Form(None, description="Optional traffic decree ID") 
):
    """Perform feature matching between two schematic maps with optional georeferencing."""
    
    # Validate inputs
    validate_common_inputs(overlay_transparency, output_format, source_image)
    ValidationUtils.validate_image_file(destination_image, "Destination image")
    ValidationUtils.validate_pgw_file(destination_pgw)
    
    # Create session
    session_id = session_manager.create_session_id(traffic_decree_id)
    session_upload_dir, session_output_dir = session_manager.setup_session_directories(session_id)
    
    try:
        # Save uploaded files
        source_path = file_manager.save_uploaded_file(source_image, session_id, "source")
        dest_path = file_manager.save_uploaded_file(destination_image, session_id, "destination")
        
        # Save PGW file if provided
        pgw_path = file_manager.save_pgw_file(destination_pgw, dest_path)
        
        logger.info(f"Processing feature matching for session {session_id}")
        logger.info(f"Source: {source_image.filename}, Destination: {destination_image.filename}")
        if destination_pgw:
            logger.info(f"PGW file provided: {destination_pgw.filename}")
        
        # Perform feature matching
        match_result = feature_matching_service.perform_feature_matching(
            source_image_path=source_path,
            destination_image_path=dest_path,
            output_dir=session_output_dir,
            overlay_transparency=overlay_transparency
        )
        
        if not match_result["success"]:
            raise HTTPException(
                status_code=422, 
                detail=f"Feature matching failed: {match_result['error_message']}"
            )
        
        # Handle response format
        if output_format == "json":
            response_data = ResponseFormatter.create_feature_matching_response(
                session_id=session_id,
                match_result=match_result,
                overlay_transparency=overlay_transparency,
                pgw_provided=destination_pgw is not None
            )
            return JSONResponse(content=response_data)
        
        else:  # output_format == "files"
            overlay_file = os.path.join(session_output_dir, "warped_overlay_result.png")
            if os.path.exists(overlay_file):
                return FileResponse(
                    overlay_file,
                    media_type="image/png",
                    filename=f"matched_overlay_{session_id}.png"
                )
            else:
                raise HTTPException(status_code=500, detail="Output file not generated")
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error in feature matching: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up uploaded files (keep outputs for download)
        file_manager.cleanup_session_uploads(session_id)


@app.post("/cut-out-georeferenced-map")
async def cut_osm_map_endpoint(
    geometry: str = Form(..., description="Geometry as GeoJSON, WKT, or coordinate list JSON string"),
    map_type: str = Form("osm", description="Map type (see endpoint documentation for full list of supported types)"),
    buffer: float = Form(800, description="Buffer distance in meters around the geometry"),
    output_format: str = Form("json", description="Output format: 'json' or 'files'"), 
    traffic_decree_id: str = Form(None, description="Optional traffic decree ID") 
):
    """Cut out a georeferenced map section from various map sources based on geometry input."""
    
    # Validate inputs
    ValidationUtils.validate_output_format(output_format)
    ValidationUtils.validate_buffer(buffer)
    ValidationUtils.validate_map_type(map_type)
    
    # Parse geometry
    geometry_input = ValidationUtils.parse_geometry_input(geometry)
    
    # Create session
    session_id = session_manager.create_session_id(traffic_decree_id)
    _, session_output_dir = session_manager.setup_session_directories(session_id)
    
    try:
        logger.info(f"Processing map cutting ({map_type}) with geometry input for session {session_id}")
        logger.info(f"Buffer: {buffer}m")
        
        # Perform map cutting
        cut_result = map_cutting_service.cut_georeferenced_map(
            geometry_input=geometry_input,
            map_type=map_type,
            buffer_meters=buffer,
            output_dir=session_output_dir,
            target_width=2048,
            output_name="temp"  # Will be overridden by descriptive name
        )
        
        if not cut_result["success"]:
            raise HTTPException(
                status_code=422, 
                detail=f"Map cutting failed: {cut_result['error_message']}"
            )
        
        # Handle response format
        if output_format == "json":
            response_data = ResponseFormatter.create_map_cutting_response(
                session_id=session_id,
                cut_result=cut_result,
                buffer_meters=buffer,
                map_type=map_type,
                target_width=2048
            )
            return JSONResponse(content=response_data)
        
        else:  # output_format == "files"
            map_file = os.path.join(session_output_dir, cut_result["files"]["map_image"])
            if os.path.exists(map_file):
                return FileResponse(
                    map_file,
                    media_type="image/png",
                    filename=f"{cut_result['output_name']}_{session_id}.png"
                )
            else:
                raise HTTPException(status_code=500, detail="Output file not generated")
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error in map cutting: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/cutout-and-match")
async def cutout_and_match(
    source_image: UploadFile = File(..., description="Source image to be warped and matched"),
    geometry: str = Form(..., description="Geometry as GeoJSON, WKT, or coordinate list JSON string for map cutting"),
    map_type: str = Form(..., description="Map type (see endpoint documentation for full list of supported types)"),
    overlay_transparency: float = Form(0.6, description="Overlay transparency (0.0-1.0)"),
    output_format: str = Form("json", description="Output format: 'json' or 'files'"),
    traffic_decree_id: str = Form(None, description="Optional traffic decree ID") 
):
    """Cut out a map section and perform feature matching with automatic buffer optimization."""
    
    # Validate inputs
    validate_common_inputs(overlay_transparency, output_format, source_image)
    ValidationUtils.validate_map_type(map_type)
    
    # Parse geometry
    geometry_input = ValidationUtils.parse_geometry_input(geometry)
    
    # Get buffer sizes to test based on map type
    test_buffer_sizes = map_cutting_service.get_buffer_sizes_for_map_type(map_type)
    
    # Create session
    session_id = session_manager.create_session_id(traffic_decree_id)
    session_upload_dir, session_output_dir = session_manager.setup_session_directories(session_id)
    
    try:
        # Save source image
        source_path = file_manager.save_uploaded_file(source_image, session_id, "source")
        
        logger.info(f"Processing cutout-and-match for session {session_id}")
        logger.info(f"Source: {source_image.filename}, Map type: {map_type}")
        
        # Test multiple buffer sizes and find the best one
        buffer_test_result = feature_matching_service.test_multiple_buffers(
            source_image_path=source_path,
            geometry_input=geometry_input,
            map_type=map_type,
            test_buffer_sizes=test_buffer_sizes,
            overlay_transparency=overlay_transparency,
            session_output_dir=session_output_dir,
            map_cutting_service=map_cutting_service
        )
        
        if not buffer_test_result["success"]:
            raise HTTPException(
                status_code=422, 
                detail=buffer_test_result["error_message"]
            )
        
        # Copy best results to final output directory
        feature_matching_service.copy_best_results(buffer_test_result["best_result"], session_output_dir)
        
        # Handle response format
        if output_format == "json":
            response_data = ResponseFormatter.create_cutout_and_match_response(
                session_id=session_id,
                match_result=buffer_test_result["best_result"]["match_result"],
                cut_result=buffer_test_result["best_result"]["cut_result"],
                overlay_transparency=overlay_transparency,
                map_type=map_type,
                best_buffer=buffer_test_result["best_buffer"],
                test_buffer_sizes=test_buffer_sizes,
                buffer_results=buffer_test_result["buffer_results"]
            )
            return JSONResponse(content=response_data)
        
        else:  # output_format == "files"
            tif_file = os.path.join(session_output_dir, "warped_source.tif")
            if os.path.exists(tif_file):
                return FileResponse(
                    tif_file,
                    media_type="image/tiff",
                    filename=f"warped_source_{session_id}.tif"
                )
            else:
                raise HTTPException(status_code=500, detail="GeoTIFF file not generated")
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error in cutout-and-match: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        file_manager.cleanup_session_uploads(session_id)


@app.get("/download/{session_id}/{file_path:path}")
async def download_file(session_id: str, file_path: str):
    """Download processed files by session ID and file path (supports subdirectories)."""
    try:
        # Validate and get full file path
        full_file_path = file_manager.validate_file_path_security(session_id, file_path)
        
        # Get filename and media type
        filename = os.path.basename(file_path)
        media_type = file_manager.get_media_type(filename)
        
        return FileResponse(full_file_path, media_type=media_type, filename=filename)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in file download: {str(e)}")
        raise HTTPException(status_code=500, detail="File download failed")


@app.delete("/sessions/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up a specific session's files."""
    try:
        ValidationUtils.validate_session_id(session_id)
        success = file_manager.cleanup_session_outputs(session_id)
        
        if success:
            response_data = ResponseFormatter.create_cleanup_response(session_id, True)
            return JSONResponse(content=response_data)
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup session: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    try:
        sessions = file_manager.list_all_sessions()
        response_data = ResponseFormatter.create_session_list_response(sessions)
        return JSONResponse(content=response_data)
    
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@app.post("/cutout-and-match-with-url")
async def cutout_and_match_with_url(
    image_url: str = Form(..., description="URL of the source image to be warped and matched"),
    geometry: str = Form(..., description="Geometry as GeoJSON, WKT, or coordinate list JSON string for map cutting"),
    map_type: str = Form(..., description="Map type (see endpoint documentation for full list of supported types)"),
    overlay_transparency: float = Form(0.6, description="Overlay transparency (0.0-1.0)"),
    output_format: str = Form("json", description="Output format: 'json' or 'files'"),
    traffic_decree_id: str = Form(None, description="Optional traffic decree ID") 
):
    """Cut out a map section and perform feature matching with a source image from URL."""
    
    # Validate inputs
    ValidationUtils.validate_output_format(output_format)
    ValidationUtils.validate_transparency(overlay_transparency)
    ValidationUtils.validate_map_type(map_type)
    ValidationUtils.validate_image_url(image_url)
    
    # Parse geometry
    geometry_input = ValidationUtils.parse_geometry_input(geometry)
    
    # Get buffer sizes to test based on map type
    test_buffer_sizes = map_cutting_service.get_buffer_sizes_for_map_type(map_type)
    
    # Create session
    session_id = session_manager.create_session_id(traffic_decree_id)
    session_upload_dir, session_output_dir = session_manager.setup_session_directories(session_id)
    
    try:
        # Download source image from URL
        source_path = file_manager.download_image_from_url(image_url, session_id, "source")
        
        logger.info(f"Processing cutout-and-match-with-url for session {session_id}")
        logger.info(f"Source URL: {image_url}, Map type: {map_type}")
        
        # Test multiple buffer sizes and find the best one
        buffer_test_result = feature_matching_service.test_multiple_buffers(
            source_image_path=source_path,
            geometry_input=geometry_input,
            map_type=map_type,
            test_buffer_sizes=test_buffer_sizes,
            overlay_transparency=overlay_transparency,
            session_output_dir=session_output_dir,
            map_cutting_service=map_cutting_service
        )
        
        if not buffer_test_result["success"]:
            raise HTTPException(
                status_code=422, 
                detail=buffer_test_result["error_message"]
            )
        
        # Copy best results to final output directory
        feature_matching_service.copy_best_results(buffer_test_result["best_result"], session_output_dir)
        
        # Handle response format
        if output_format == "json":
            response_data = ResponseFormatter.create_cutout_and_match_response(
                session_id=session_id,
                match_result=buffer_test_result["best_result"]["match_result"],
                cut_result=buffer_test_result["best_result"]["cut_result"],
                overlay_transparency=overlay_transparency,
                map_type=map_type,
                best_buffer=buffer_test_result["best_buffer"],
                test_buffer_sizes=test_buffer_sizes,
                buffer_results=buffer_test_result["buffer_results"]
            )
            return JSONResponse(content=response_data)
        
        else:  # output_format == "files"
            tif_file = os.path.join(session_output_dir, "warped_source.tif")
            if os.path.exists(tif_file):
                return FileResponse(
                    tif_file,
                    media_type="image/tiff",
                    filename=f"warped_source_{session_id}.tif"
                )
            else:
                raise HTTPException(status_code=500, detail="GeoTIFF file not generated")
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error in cutout-and-match-with-url: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        file_manager.cleanup_session_uploads(session_id)

if __name__ == "__main__":
    # Run the server with dynamic port for cloud deployment
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "app_refactored:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )