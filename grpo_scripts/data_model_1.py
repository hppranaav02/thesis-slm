import pydantic
from typing import List, Optional
from pydantic import BaseModel, Field

class ResponseFormat(BaseModel):
    type: str = Field(..., description="json")
    vulnerability: bool = Field(..., description="Whether the response should includess a vulnerability")
    vulnerability_type: Optional[str] = Field(None, description="Type of vulnerability.")
    reasoning: Optional[str] = Field(None, description="Reasoning for the response")
    source: Optional[str] = Field(None, description="Line of code where the vulnerability is found")

instruction = (
    "Classify the Go code as Vulnerable or Secure, provide the CWE ID if applicable, "
    "and return the response in the following format:\n"
    f"{ResponseFormat.schema_json(indent=2)}"
).join('')