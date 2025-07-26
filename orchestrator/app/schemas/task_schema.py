from pydantic import BaseModel
from typing import Dict, Any

class TaskRequest(BaseModel):
    task_description: str
