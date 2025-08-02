from anthropic import BaseModel
from fastapi import APIRouter

from app.controllers.InvokeController import InvokeController

router = APIRouter()

class InvokeInput(BaseModel):
    user_input: str


@router.post("/invoke")
async def invoke(user_input: InvokeInput):

    controller = InvokeController()
    response = await controller.invoke(user_input=user_input)

    return {
        "Agent Response": response,
    }