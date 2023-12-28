from typing import Any, Optional, Union
from pydantic import BaseModel

class PredictModel(BaseModel):
    modelname: str
    data: Any