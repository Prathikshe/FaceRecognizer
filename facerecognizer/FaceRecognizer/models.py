from pydantic import BaseModel

class ErrorResponse(BaseModel):
    status: str
    status_code: str
    message: str
    result: dict

class Payload_recognize_face(BaseModel):
    count: int
    profiles: list[dict]

class SuccessResponse(BaseModel):
    status: str
    status_code: str
    message: str
    result: Payload_recognize_face

class Payload_added_image(BaseModel):
    profile: str
    new_image: str

class Payload_add_image(BaseModel):
    status: str
    status_code: str
    message: str
    result: Payload_added_image

class Profile_list(BaseModel):
    profile_list: list[str]

class Payload_list_folders(BaseModel):
    status: str
    status_code: str
    message: str
    result: Profile_list

class Profile_image_list(BaseModel):
    images_list: list[str]

class Payload_image_list(BaseModel):
    status: str
    status_code: str
    message: str
    result: Profile_image_list

class DeleteResponse(BaseModel):
    status: str
    status_code: str
    message: str
    result: dict