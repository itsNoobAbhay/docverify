from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pathlib import Path
import easyocr
import re
import os
import uuid
import shutil
import numpy as np
import cv2

# Create app
app = FastAPI(title="Document Verification API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        return f.read()

@app.post("/verify/aadhar")
async def verify_aadhar(file: UploadFile = File(...)):
    """
    Verify Aadhar card and extract details
    """
    # Save file temporarily
    file_path = await save_upload_file(file)
    
    try:
        # Read the image
        image = cv2.imread(str(file_path))
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get OCR results
        results = reader.readtext(image)
        
        # Extract text from OCR results
        text = " ".join([result[1] for result in results])
        
        # Check for Aadhar card pattern (12 digits, possibly with spaces)
        aadhar_pattern = r'\b\d{4}\s?\d{4}\s?\d{4}\b'
        aadhar_match = re.search(aadhar_pattern, text)
        
        if aadhar_match:
            aadhar_number = aadhar_match.group(0).replace(" ", "")
            return {
                "valid": True,
                "document_type": "Aadhar Card",
                "aadhar_number": aadhar_number,
                "confidence": "high" if len(text) > 50 else "medium"
            }
        else:
            return {
                "valid": False,
                "document_type": "Unknown",
                "message": "No valid Aadhar number found"
            }
    except Exception as e:
        return {"valid": False, "error": str(e)}
    finally:
        # Clean up
        if file_path.exists():
            os.remove(file_path)


     
@app.post("/verify/pan")
async def verify_pan(file: UploadFile = File(...)):
    """
    Verify PAN card and extract details with improved OCR and error handling
    """
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        print(f"Processing file: {file_path}")

        # Image preprocessing pipeline - try multiple approaches
        results = []
        extracted_texts = []
        
        # Approach 1: Basic grayscale with adaptive thresholding
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh1 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        denoised1 = cv2.fastNlMeansDenoising(thresh1, None, 10, 7, 21)
        results1 = reader.readtext(denoised1)
        text1 = " ".join([result[1] for result in results1])
        extracted_texts.append(text1)
        
        # Approach 2: Different thresholding parameters
        thresh2 = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 8
        )
        results2 = reader.readtext(thresh2)
        text2 = " ".join([result[1] for result in results2])
        extracted_texts.append(text2)
        
        # Approach 3: Direct OCR on grayscale with sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        results3 = reader.readtext(sharpened)
        text3 = " ".join([result[1] for result in results3])
        extracted_texts.append(text3)
        
        # Approach 4: Try on original image
        results4 = reader.readtext(image)
        text4 = " ".join([result[1] for result in results4])
        extracted_texts.append(text4)
        
        # Combine all results
        text = " ".join(extracted_texts)
        print("\nExtracted Text from PAN Card:\n", text, "\n")
        
        # Improved PAN number pattern (more flexible)
        pan_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'
        dob_pattern = r'\b\d{2}/\d{2}/\d{4}\b'
        alt_dob_pattern = r'\b\d{2}-\d{2}-\d{4}\b'  # Alternative date format
        
        # Extract PAN Number with multiple strategies
        pan_number = None
        
        # Strategy 1: Direct regex search
        pan_matches = re.findall(pan_pattern, text)
        if pan_matches:
            pan_number = pan_matches[0]
            print(f"PAN found with direct regex: {pan_number}")
        
        # Strategy 2: Search with reduced spacing
        if not pan_number:
            no_space_text = "".join(text.split())
            pan_matches = re.findall(pan_pattern, no_space_text)
            if pan_matches:
                pan_number = pan_matches[0]
                print(f"PAN found with no-space text: {pan_number}")
        
        # Strategy 3: Look for context clues
        if not pan_number:
            context_patterns = [
                r'(?:Permanent Account Number|PAN|P\.A\.N)[:\s]*([A-Z0-9]{10})',
                r'(?:Number|No|Card No)[:\s]*([A-Z0-9]{10})'
            ]
            
            for pattern in context_patterns:
                matches = re.search(pattern, text, re.IGNORECASE)
                if matches and re.match(pan_pattern, matches.group(1)):
                    pan_number = matches.group(1)
                    print(f"PAN found with context: {pan_number}")
                    break
        
        # Extract DOB using multiple patterns
        dob = None
        dob_match = re.search(dob_pattern, text)
        if dob_match:
            dob = dob_match.group(0)
        else:
            alt_dob_match = re.search(alt_dob_pattern, text)
            if alt_dob_match:
                dob = alt_dob_match.group(0)
        
        if not dob:
            # Look for context clues for DOB
            dob_context_patterns = [
                r'(?:Date of Birth|DOB|D\.O\.B)[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})',
                r'(?:Birth|Born)[:\s]*(\d{2}[/-]\d{2}[/-]\d{4})'
            ]
            
            for pattern in dob_context_patterns:
                matches = re.search(pattern, text, re.IGNORECASE)
                if matches:
                    dob = matches.group(1)
                    break
        
        # Extract Name and Father's Name with improved context recognition
        name, father_name = None, None
        
        # Look for name with context
        name_patterns = [
            r'(?:Name|NAME)[:\s]*([A-Za-z\s]+)',
            r'(?:Name of the PAN card holder|Holder\'s Name)[:\s]*([A-Za-z\s]+)'
        ]
        
        for pattern in name_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                candidate = matches.group(1).strip()
                if len(candidate.split()) <= 5 and len(candidate) >= 3:
                    name = candidate
                    break
        
        # Look for father's name with context
        father_patterns = [
            r'(?:Father\'s Name|FATHER|Father)[:\s]*([A-Za-z\s]+)',
            r'(?:Father\'s|Father|F/O)[:\s]*([A-Za-z\s]+)'
        ]
        
        for pattern in father_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                candidate = matches.group(1).strip()
                if len(candidate.split()) <= 5 and len(candidate) >= 3:
                    father_name = candidate
                    break
        
        # Determine validity and confidence
        pan_valid = bool(pan_number) and len(pan_number) == 10
        indicators = [
            "INCOME TAX" in text.upper(),
            "GOVT OF INDIA" in text.upper() or "GOVERNMENT OF INDIA" in text.upper(),
            "PERMANENT ACCOUNT NUMBER" in text.upper() or "PAN" in text.upper(),
            bool(dob),
            bool(name)
        ]
        
        # Calculate confidence score (0-1)
        confidence_score = sum(1 for ind in indicators if ind) / len(indicators)
        
        # Map confidence score to category
        if confidence_score > 0.7:
            confidence = "high"
        elif confidence_score > 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        print(f"Detected PAN Number: {pan_number}")
        print(f"Detected DOB: {dob}")
        print(f"Extracted Name: {name}")
        print(f"Extracted Father's Name: {father_name}")
        print(f"Confidence: {confidence} ({confidence_score:.2f})")
        
        return {
            "valid": pan_valid,
            "document_type": "PAN Card",
            "pan_number": pan_number if pan_number else "Not Found",
            "name": name if name else "Not Found",
            "father_name": father_name if father_name else "Not Found",
            "date_of_birth": dob if dob else "Not Found",
            "confidence": confidence,
            "confidence_score": round(confidence_score, 2)
        }

    except Exception as e:
        print(f"Error in PAN verification: {str(e)}")
        traceback.print_exc()  # Print full traceback for debugging
        return {"valid": False, "error": str(e)}

    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)



async def save_upload_file(upload_file: UploadFile) -> Path:
    """
    Save uploaded file to a temporary location
    """
    # Create a random filename to avoid collisions
    file_name = f"{uuid.uuid4()}{os.path.splitext(upload_file.filename)[1]}"
    file_path = UPLOAD_DIR / file_name
    
    # Save the file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)