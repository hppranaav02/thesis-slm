from typing import List, Optional, Union
from pydantic import BaseModel, Field
import json
class ResponseFormat(BaseModel):
    type: str = Field(..., description="json")
    vulnerability: bool = Field(
        ..., description="Whether the response includes a vulnerability"
    )
    vulnerability_type: Optional[str] = Field(
        None, description="Single CWE ID (e.g. 'CWE-178')"
    )
    reasoning: Optional[str] = Field(None, description="Reasoning for the response")
    source: Optional[str] = Field(
        None, description="Line of code where the vulnerability is found"
    )


class ResponseFormatMulti(BaseModel):
    type: str = Field(..., description="json")
    vulnerability: bool = Field(
        ..., description="Whether the response includes a vulnerability"
    )
    vulnerability_type: Optional[List[str]] = Field(
        None,
        description="List of CWE IDs, e.g. ['CWE-178', 'CWE-863'] "
        "(empty or omitted if secure)",
    )
    reasoning: Optional[str] = Field(None, description="Reasoning for the response")
    source: Optional[Union[str, List[str]]] = Field(
        None, description="Line(s) of code where the vulnerability is found"
    )

instruction_single = (
    "Classify the Go code as Vulnerable or Secure, provide the CWE ID if applicable, "
    "think step by step, and always return reasoning and the line of code as source.\n"
    + ResponseFormat.schema_json(indent=2)
)

response_format_dict = {
    "vulnerability": "<boolean>",
    "vulnerability_type": "<list of CWE IDs or empty>",
    "reasoning": "optional[<string>]",
    "source": "optional[<line(s) of code where the vulnerability is found>]"
}

instruction_multi = (
    "Classify the Go code as Vulnerable or Secure, provide *all* CWE IDs if applicable "
    "as a JSON array, think step by step, and always return reasoning for the choice and source "
    "line(s).\n ```json\n"
    + json.dumps(response_format_dict, indent=2) + "\n```"
)


if __name__ == "__main__":
    print("Single-label instruction:")
    print(instruction_single)
    print("\nMulti-label instruction:")
    print(instruction_multi)
    print("\nResponse format dictionary:")
    print(response_format_dict)
    print("\nSingle-label schema JSON:")
    print(ResponseFormat.schema_json(indent=2))
    print("\nMulti-label schema JSON:")
    print(ResponseFormatMulti.schema_json(indent=2))