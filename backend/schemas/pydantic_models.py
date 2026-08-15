from pydantic import BaseModel, Field
from typing import List, Optional

class UploadResponse(BaseModel):
    filename : str = Field(... , description = "The name of the file")
    message : str = Field(... , description = "A message whether the file is received properly or not")
    document_id : str = Field(... , description = "The document ID of the uploaded file")

class ChatBot(BaseModel):
    role : str = Field(... , description="The role of the speaker ('user' or 'assistant')")
    content : str = Field(... , description = "The text content of the message")

class QueryRequest(BaseModel):
    query: str = Field(..., description="The query", min_length=2)
    top_k: int = Field(3, description="Number of results to retrieve")
    chat_history: Optional[List[ChatBot]] = Field(default=[], description="Optional conversational history")
    source_file: Optional[str] = Field(default=None, description="Optional PDF filename filter")

class Citation(BaseModel):
    source_file: str = Field(..., description="The name of the PDF this information came from")
    page_number: int = Field(..., description="The exact page the text was found on.")
    text_snippet: str = Field(..., description="A short preview of the exact paragraph the AI used.")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer by the LLM for the query")
    citations: List[Citation] = Field(..., description="A list of citations to support the answer")


