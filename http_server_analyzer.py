from flask import Flask, request, jsonify
import sys
import os
import base64
import numpy as np
import cv2
import json
from werkzeug.utils import secure_filename
import tempfile

# Add the Python scripts directory to the Python path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))

# For local development
python_scripts_dir = os.path.join(current_dir, "Assets", "Scripts", "Python")
if os.path.exists(python_scripts_dir):
    sys.path.append(python_scripts_dir)
    print(f"Added to Python path: {python_scripts_dir}")
else:
    # For production (files are in root directory)
    print("Using production mode - importing from root directory")

# Import our image analysis modules
try:
    from Image_analyzer import run_detection # type: ignore
    from testing import run_detection as run_detection_testing
    from CircleDetector import circle_detector # type: ignore
    from ImageClearer import image_clearer # type: ignore
    print("Successfully imported image analysis modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Available files in current directory:")
    for file in os.listdir('.'):
        if file.endswith('.py'):
            print(f"  {file}")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Add CORS headers for cross-origin requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def process_image_from_array(image_array):
    """Process image from numpy array and return analysis results"""
    try:
        # Ensure the image is in the right format
        if len(image_array.shape) == 3:
            # Convert BGR to RGB if needed (OpenCV loads as BGR)
            image_bgr = image_array
        else:
            image_bgr = image_array
        
        # Save the image temporarily as the run_detection function expects a file path
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, image_bgr)
        
        try:
            # Call the main analysis function
            print("Starting image analysis...")
            
            # Use the run_detection function from our Image_analyzer module
            # Run both analyzers
            results_image = run_detection(temp_path)
            results_testing = run_detection_testing(temp_path)

            results = {
                "image_analyzer": results_image,
                "testing_analyzer": results_testing
            }

            
            print(f"Analysis completed: {results}")
            return results
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return {"error": f"Image processing failed: {str(e)}"}

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    import time
    return jsonify({
        "status": "healthy",
        "message": "Image Analysis Server is running",
        "version": "1.0",
        "analyzer_available": True,
        "timestamp": time.time()
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify server functionality"""
    return jsonify({
        "status": "success",
        "message": "Server is working correctly",
        "available_endpoints": ["/health", "/test", "/analyze"]
    })

@app.route('/analyze', methods=['POST'])
def analyze_image_endpoint():
    """Main endpoint for image analysis"""
    try:
        print(f"Received analysis request. Content-Type: {request.content_type}")
        print(f"Request files: {list(request.files.keys())}")
        print(f"Request form: {list(request.form.keys())}")
        print(f"Request json available: {request.is_json}")
        
        image_array = None
        
        # Method 1: Handle multipart/form-data with file upload
        if 'image' in request.files:
            print("Processing uploaded file...")
            file = request.files['image']
            if file.filename == '':
                return jsonify({"success": False, "error": "No file selected"}), 400
            
            # Read the file into memory
            file_data = file.read()
            
            # Convert to numpy array
            nparr = np.frombuffer(file_data, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image_array is None:   
                return jsonify({"success": False, "error": "Could not decode image file"}), 400
                
            print(f"Decoded image shape: {image_array.shape}")
            
        # Method 2: Handle JSON with base64 encoded image
        elif request.is_json:
            print("Processing JSON request...")
            data = request.get_json()
            
            if 'image_data' in data:
                try:
                    # Decode base64 image
                    image_data = data['image_data']
                    
                    # Remove data URL prefix if present
                    if image_data.startswith('data:image'):
                        image_data = image_data.split(',')[1]
                    
                    # Decode base64
                    decoded_data = base64.b64decode(image_data)
                    
                    # Convert to numpy array
                    nparr = np.frombuffer(decoded_data, np.uint8)
                    image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image_array is None:
                        return jsonify({"success": False, "error": "Could not decode base64 image"}), 400
                        
                    print(f"Decoded base64 image shape: {image_array.shape}")
                    
                except Exception as e:
                    return jsonify({"success": False, "error": f"Base64 decode error: {str(e)}"}), 400
            else:
                return jsonify({"success": False, "error": "No 'image_data' field in JSON"}), 400
                
        # Method 3: Handle form data with base64
        elif 'image_data' in request.form:
            print("Processing form data...")
            try:
                image_data = request.form['image_data']
                
                # Remove data URL prefix if present
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                # Decode base64
                decoded_data = base64.b64decode(image_data)
                
                # Convert to numpy array
                nparr = np.frombuffer(decoded_data, np.uint8)
                image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image_array is None:
                    return jsonify({"success": False, "error": "Could not decode form base64 image"}), 400
                    
                print(f"Decoded form base64 image shape: {image_array.shape}")
                
            except Exception as e:
                return jsonify({"success": False, "error": f"Form base64 decode error: {str(e)}"}), 400
        else:
            return jsonify({"success": False, "error": "No image data provided. Send either: 1) multipart/form-data with 'image' file, 2) JSON with 'image_data' base64, or 3) form data with 'image_data' base64"}), 400
        
        # Process the image
        if image_array is not None:
            print("Starting image analysis...")
            results = process_image_from_array(image_array)
            
            if "error" in results:
                return jsonify({
                    "success": False,
                    "error": results["error"]
                }), 500
            
            return jsonify({
                "success": True,
                "rawJsonData": json.dumps(results),
                "results": results,
                "image_shape": list(image_array.shape)
            })
        else:
            return jsonify({
                "success": False,
                "error": "Failed to process image data"
            }), 400
            
    except Exception as e:
        print(f"Unexpected error in analyze endpoint: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Get port from environment variable (for cloud deployment) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print("Starting Image Analysis HTTP Server...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /test - Test endpoint") 
    print("  POST /analyze - Image analysis")
    print()
    print(f"Server will run on port {port}")
    
    # Run the Flask app
    if os.environ.get('FLASK_ENV') == 'production':
        # Production mode (for Render)
        app.run(host='0.0.0.0', port=port, debug=False)
    else:
        # Development mode
        app.run(host='0.0.0.0', port=port, debug=True)