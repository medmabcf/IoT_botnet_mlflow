from typing import Any
from pydantic import BaseModel

class PredictModel(BaseModel):
    model_name: str
    data: Any