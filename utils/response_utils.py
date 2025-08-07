"""
Response utilities for formatting API responses
"""
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ResponseFormatter:
    """Handles formatting of API responses"""
    
    @staticmethod
    def create_feature_matching_response(
        session_id: str,
        match_result: Dict[str, Any],
        overlay_transparency: float,
        pgw_provided: bool = False,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create standardized feature matching response
        
        Args:
            session_id: Session identifier
            match_result: Results from matching service
            overlay_transparency: Transparency used
            pgw_provided: Whether PGW file was provided
            additional_info: Additional processing info
            
        Returns:
            Formatted response dictionary
        """
        # Convert georeferenced files to URLs
        georeferenced_files = []
        if match_result.get("georeferenced_files"):
            for filename in match_result["georeferenced_files"]:
                # Determine file type from extension
                if isinstance(filename, tuple):
                    # If it's already a tuple (file_type, filename)
                    file_type, filename = filename
                else:
                    # If it's just a filename, determine type from extension
                    if filename.lower().endswith(('.tif', '.tiff')):
                        file_type = "geotiff"
                    elif filename.lower().endswith('.pgw'):
                        file_type = "world_file"
                    elif filename.lower().endswith('.prj'):
                        file_type = "projection"
                    else:
                        file_type = "unknown"
                georeferenced_files.append({
                    "type": file_type,
                    "url": f"/download/{session_id}/{filename}"
                })
        
        response_data = {
            "success": True,
            "session_id": session_id,
            "processing_info": {
                "matches_count": match_result.get("matches_count", 0),
                "inlier_ratio": match_result.get("inlier_ratio", 0.0),
                "overlay_transparency": overlay_transparency,
                "georeferenced": match_result.get("georeferenced", False),
                "pgw_provided": pgw_provided
            },
            "output_files": {
                "feature_matching": {
                    "all_matches": f"/download/{session_id}/all_feature_matches.png",
                    "inlier_matches": f"/download/{session_id}/inlier_matches.png",
                    "outlier_matches": f"/download/{session_id}/outlier_matches.png",
                    "analysis": f"/download/{session_id}/feature_matching_analysis.png",
                    "warped_overlay": f"/download/{session_id}/warped_overlay_result.png",
                    "warped_source": f"/download/{session_id}/warped_source.png"
                }
            },
            "georeferenced_files": georeferenced_files,
            "timestamp": datetime.now().isoformat(),
            "quality_assessment": {
                "status": match_result.get("quality_status", "unknown"),
                "inlier_ratio": match_result.get("inlier_ratio", 0.0),
                "matches_count": match_result.get("matches_count", 0)
            }
        }
        
        # Add additional info if provided
        if additional_info:
            response_data["processing_info"].update(additional_info)
        
        return response_data
    
    @staticmethod
    def create_map_cutting_response(
        session_id: str,
        cut_result: Dict[str, Any],
        buffer_meters: float,
        map_type: str,
        target_width: int
    ) -> Dict[str, Any]:
        """
        Create standardized map cutting response
        
        Args:
            session_id: Session identifier
            cut_result: Results from cutting service
            buffer_meters: Buffer used
            map_type: Map type
            target_width: Target width used
            
        Returns:
            Formatted response dictionary
        """
        processing_info = {
            "buffer_meters": buffer_meters,
            "map_type": map_type,
            "target_width": target_width
        }
        
        # Add image info
        if "image_shape" in cut_result:
            processing_info["actual_size"] = {
                "width": cut_result["image_shape"]["width"],
                "height": cut_result["image_shape"]["height"]
            }
        
        # Add geometry info if available
        if "input_type" in cut_result:
            processing_info["input_type"] = cut_result["input_type"]
        if "geometry_type" in cut_result:
            processing_info["geometry_type"] = cut_result["geometry_type"]
        if "geometry_format" in cut_result:
            processing_info["geometry_format"] = cut_result["geometry_format"]
        if "point_count" in cut_result:
            processing_info["point_count"] = cut_result["point_count"]
        
        response_data = {
            "success": True,
            "session_id": session_id,
            "processing_info": processing_info,
            "bounds_rd": cut_result.get("bounds"),
            "files": {
                "map_image": f"/download/{session_id}/{cut_result['files']['map_image']}",
                "world_file": f"/download/{session_id}/{cut_result['files']['world_file']}"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return response_data
    
    @staticmethod
    def create_cutout_and_match_response(
        session_id: str,
        match_result: Dict[str, Any],
        cut_result: Dict[str, Any],
        overlay_transparency: float,
        map_type: str,
        best_buffer: float,
        test_buffer_sizes: List[int],
        buffer_results: List[Dict[str, Any]],
        source_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create response for cutout-and-match operations
        
        Args:
            session_id: Session identifier
            match_result: Results from matching
            cut_result: Results from cutting
            overlay_transparency: Transparency used
            map_type: Map type
            best_buffer: Selected buffer size
            test_buffer_sizes: All tested buffer sizes
            buffer_results: Results from all buffer tests
            source_url: Optional source URL for URL-based operations
            
        Returns:
            Formatted response dictionary
        """
        # Start with basic feature matching response
        response_data = ResponseFormatter.create_feature_matching_response(
            session_id=session_id,
            match_result=match_result,
            overlay_transparency=overlay_transparency,
            pgw_provided=True  # Always true since we generate it from map cutting
        )
        
        # Add source URL if provided
        if source_url:
            response_data["processing_info"]["source_url"] = source_url
        
        # Add map cutting information
        response_data["processing_info"]["map_cutting"] = {
            "map_type": map_type,
            "selected_buffer_meters": best_buffer,
            "bounds_rd": cut_result.get("bounds"),
            "destination_name": cut_result.get("output_name"),
            "buffer_selection": {
                "tested_buffers": test_buffer_sizes,
                "results": [
                    {
                        "buffer_meters": r["buffer_meters"],
                        "map_type": r.get("map_type", map_type),
                        "matches_count": r["matches_count"],
                        "inlier_count": r["inlier_count"],
                        "inlier_ratio": r["inlier_ratio"]
                    } for r in buffer_results
                ],
                "selection_criteria": "maximum_inlier_count",
                "inspection_folders": [
                    f"/download/{session_id}/buffer_{size}m_cutout/" 
                    for size in test_buffer_sizes
                ]
            }
        }
        
        return response_data
    
    @staticmethod
    def create_error_response(
        error_message: str,
        status_code: int = 500,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create standardized error response
        
        Args:
            error_message: Error message
            status_code: HTTP status code
            additional_info: Additional error info
            
        Returns:
            Error response dictionary
        """
        response_data = {
            "success": False,
            "error_message": error_message,
            "status_code": status_code,
            "timestamp": datetime.now().isoformat()
        }
        
        if additional_info:
            response_data.update(additional_info)
        
        return response_data
    
    @staticmethod
    def create_session_list_response(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create response for session listing
        
        Args:
            sessions: List of session info dictionaries
            
        Returns:
            Session list response
        """
        # Convert timestamps to ISO format
        formatted_sessions = []
        for session in sessions:
            formatted_session = session.copy()
            if "created" in formatted_session:
                formatted_session["created"] = datetime.fromtimestamp(
                    formatted_session["created"]
                ).isoformat()
            formatted_sessions.append(formatted_session)
        
        return {
            "sessions": formatted_sessions,
            "count": len(formatted_sessions),
            "timestamp": datetime.now().isoformat()
        }
    
    @staticmethod
    def create_cleanup_response(session_id: str, success: bool = True) -> Dict[str, Any]:
        """
        Create response for session cleanup
        
        Args:
            session_id: Session identifier
            success: Whether cleanup was successful
            
        Returns:
            Cleanup response
        """
        if success:
            return {
                "message": f"Session {session_id} cleaned up successfully",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "error": f"Session {session_id} not found",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }