from deepface import DeepFace
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Security, Depends
from fastapi.responses import JSONResponse
import shutil
import logging
from pathlib import Path
from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime, timedelta
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from models import (
    Payload_recognize_face,
    SuccessResponse,
    ErrorResponse,
    Payload_added_image,
    Payload_add_image,
    Profile_list,
    Payload_list_folders,
    Profile_image_list,
    Payload_image_list,
    DeleteResponse
)
import json

security = HTTPBearer()

def get_secret_key():
    with open("setting.json", "r") as file:
        settings = json.load(file)
        
    api_key = settings.get("jwt_private_key")
    return api_key

SECRET_KEY = get_secret_key()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return True
    except:
        return JSONResponse(status_code=404, content=ErrorResponse(
                status="ERROR",
                status_code="404",
                message="Invalid authorization token",
                result={}
                ).dict())

description = "The FACE_RECOGNIZER API allows users to perform face recognition tasks efficiently. It provides endpoints to upload reference images, recognize faces, list folders and images, and delete images. The API is ideal for systems requiring facial identification or comparison functionality."

app = FastAPI(
    title="FACE_RECOGNIZER",
    description=description,
    summary="Face recognize using deepface",
    version="1.0.0",
    contact={
        "name": "Prathik",
        "url": "https://github.com/Prathikshe",
        "email": "prathikpshet21@gmail.com",
    },
    docs_url="/docs",
    root_path="/face_recognizer",
    redoc_url=None,)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "test_imgs")
folder_to_search = os.path.join(BASE_DIR, "face_list")

os.makedirs(image_dir, exist_ok=True)

# Function to calculate confidence
def calculate_confidence(row):
    return max(0, (row["threshold"] - row["distance"]) / row["threshold"]) * 100

# Endpoint to receive an image and perform face recognition
@app.post("/recognize", response_model=SuccessResponse, summary="Face recognizer")
async def recognize_face(image: UploadFile = File(...), authorized: bool = Depends(verify_token)):
    if authorized:   
        image_to_find = None
        try:
            if not image:
                logging.info("400 - No image provided in the request")
                raise HTTPException(status_code=400, detail="No image provided")

            # Retrieve the uploaded image's filename (without extension)
            filename = os.path.splitext(image.filename)[0]

            # Save the uploaded image with the same name as in the request
            image_to_find = os.path.join(image_dir, image.filename)
            with open(image_to_find, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)

            # Define the folder to search and distance metric
            metric_model = "euclidean_l2"
            if not os.path.exists(folder_to_search):
                raise HTTPException(status_code=400, detail="Faces list folder is missing")
            elif not any(Path(folder_to_search).iterdir()):  # Check if folder is empty
                raise HTTPException(status_code=400, detail="Faces list folder is empty")

            try:
                face_objs = DeepFace.extract_faces(img_path=image_to_find,anti_spoofing=True)
                if face_objs:  # Check if the list is not empty
                    is_real = face_objs[0].get('is_real', None)  # Safely get 'is_real' from the first face object
 
            # Perform face recognition
            try:
                face_objs = DeepFace.extract_faces(img_path=image_to_find, anti_spoofing=True)

                if face_objs:  # Check if the list is not empty
                    is_real = face_objs[0].get('is_real', None)  # Safely get 'is_real' from the first face object
                    
                    if is_real:  # Check if the detected face is real
                        dfs = DeepFace.find(
                            img_path=image_to_find, 
                            db_path=folder_to_search, 
                            distance_metric=metric_model, 
                            anti_spoofing=True
                        )
                    else:  # Handle spoofing case
                        logging.error("Not a real face or photo face recognized.")   
                        return JSONResponse(status_code=404, content=ErrorResponse(
                            status="ERROR",
                            status_code="404",
                            message="Spoofing face image found",
                            result={}
                            ).dict())   
            except Exception as e:
                logging.error(f"Error during face recognition: {str(e)}")
                return JSONResponse(status_code=404, content=ErrorResponse(
                            status="ERROR",
                            status_code="404",
                            message="Match not found",
                            result={}
                            ).dict())

            # Check if the result is empty
            if isinstance(dfs, list) and len(dfs) == 0:
                response= SuccessResponse(
                    status="OK",
                    status_code="200",
                    message= "Match not found",
                    result= {}
                )
                
            elif isinstance(dfs, list) and len(dfs) > 0:
                matches = dfs[0]
                # Add confidence to the DataFrame
                matches["confidence"] = matches.apply(calculate_confidence, axis=1)
                profiles = []
                for _, row in matches.iterrows():
                    matched_image_path = row["identity"]
                    matched_image_folder = Path(matched_image_path).parent.name
                    matched_image_name = os.path.splitext(os.path.basename(matched_image_path))[0]
                    confidence = f"{row['confidence']:.2f}"  # Format confidence to 2 decimal places

                    profiles.append({
                        "profile_name": matched_image_folder,
                        "confidence": confidence
                    })

                response= SuccessResponse(
                    status="OK",
                    status_code="200",
                    message= "",
                    result= Payload_recognize_face(
                        count= len(profiles),
                        profiles= profiles
                    )
                )
            else:
                response= SuccessResponse(
                    status="OK",
                    status_code="400",
                    message= "Error: Unknown response format",
                    result= {}
                )           
            return response

        except Exception as e:
            logging.error(f"Error processing the request: {str(e)}")
            return JSONResponse(status_code=404, content=ErrorResponse(
                status="ERROR",
                status_code="404",
                message="Uploaded file is not an image Or bad request",
                result={}
                ).dict())

        finally:
            if os.path.exists(image_to_find):
                os.remove(image_to_find)

@app.put("/profile", response_model=Payload_add_image, summary="Add new profile or face image to existing profile")
async def add_reference_images(profile: str, single_file: UploadFile = File(...), authorized: bool = Depends(verify_token)):
    if authorized:
        try:
            # Check if the uploaded file is an image
            if not single_file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded file is not an image."
                )
            
            # Generate a new file name with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{profile}_{timestamp}{Path(single_file.filename).suffix}"
            
            # Create profile folder if it doesn't exist
            folder_path = Path(folder_to_search) / profile
            folder_path.mkdir(parents=True, exist_ok=True)
            
            # Save the file to the designated folder
            file_path = folder_path / new_filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(single_file.file, buffer)
            
            # Return success response
            response = Payload_add_image(
                status="OK",
                status_code="200",
                message="Image uploaded successfully",
                result=Payload_added_image(
                    profile=profile,
                    new_image=new_filename
                )
            )
            return response
        except Exception as e:
            logging.error(f"Error: {str(e.detail)}")
            raise HTTPException(status_code=404, detail=ErrorResponse(
                status="ERROR",
                status_code="404",
                message="Uploaded file is not an image.",
                result={}
                ).dict())

@app.get("/profile", response_model=Payload_list_folders, summary="List profiles")
async def list_folders(authorized: bool = Depends(verify_token)):
    if authorized:
        folders = [folder.name for folder in Path(folder_to_search).iterdir() if folder.is_dir()]
        response = Payload_list_folders(
            status="OK",
            status_code="200",  
            message="",
            result = Profile_list(
                profile_list=folders
            )
        )
        return response

@app.get("/profile/{profile_name}", response_model=Payload_image_list, summary="List face images in a profile")
async def list_images(profile_name: str, authorized: bool = Depends(verify_token)):
    if authorized:
        try:
            folder_path = Path(folder_to_search) / profile_name
            if not folder_path.exists() or not folder_path.is_dir():
                raise HTTPException(status_code=404, detail=f"Folder '{profile_name}' not found")
            images = [image.name for image in folder_path.iterdir() if image.is_file()]
            response = Payload_image_list(
                status="OK",
                status_code="200",  
                message="",
                result = Profile_image_list(
                    images_list=images
                )
            )
            return response
        except HTTPException as e:
            logging.error(f"Error: {str(e.detail)}")
            raise HTTPException(status_code=404, detail= ErrorResponse(
                status="ERROR",
                status_code="404",
                message="Profile is not found",
                result={}
            ).dict())

@app.delete("/profile/{profile_name}", response_model=DeleteResponse, summary="Delete profile")
async def list_images(profile_name: str, authorized: bool = Depends(verify_token)):
    if authorized:
        folder_path = Path(folder_to_search) / profile_name
        if not folder_path.exists() or not folder_path.is_dir():
            error = ErrorResponse(
                status="ERROR",
                status_code="404",
                message="Profile is not found",
                result={}
            )
            return error
        shutil.rmtree(folder_path)
        response = DeleteResponse(
            status="OK",
            status_code="200",  
            message="Profile deleted successfully",
            result = {}
        )
        return response

@app.delete("/profile/{profile_name}/{image_name}", response_model=DeleteResponse, summary="Delete image from a profile")
async def list_images(profile_name: str, image_name: str, authorized: bool = Depends(verify_token)):
    if authorized:
        file_path = Path(folder_to_search) / profile_name / image_name
        folder_path = Path(folder_to_search) / profile_name
        if not folder_path.exists() or not folder_path.is_dir():
            error = ErrorResponse(
                status="ERROR",
                status_code="404",
                message="Profile is not found",
                result={}
            )
            return error
        if not file_path.exists():
            error = ErrorResponse(
                status="ERROR",
                status_code="404",
                message="Image name is not found",
                result={}
            )
            return error
        os.remove(file_path)
        response = DeleteResponse(
            status="OK",
            status_code="200",  
            message="Image deleted successfully",
            result = {}
        )
        return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9009)
