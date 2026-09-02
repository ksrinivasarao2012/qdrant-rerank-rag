from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal

class UploadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filename: str = Field(..., description="The name of the file")
    message: str = Field(..., description="A message whether the file is received properly or not")
    document_id: str = Field(..., description="The document ID of the uploaded file")

class ChatBot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"] = Field(..., description="The role of the speaker")
    content: str = Field(..., min_length=1, max_length=4_000, description="The text content of the message")

class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(..., min_length=2, max_length=1_000, description="The query string")
    top_k: int = Field(3, ge=1, le=20, description="Number of results to retrieve (min 1, max 20)")
    chat_history: List[ChatBot] = Field(default_factory=list, max_length=20, description="Optional conversational history (max 20 turns)")
    source_file: Optional[str] = Field(default=None, max_length=100, description="Optional tag filter (e.g. 'regression').")

class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_file: str = Field(..., description="The question title of the StackExchange post the answer came from.")
    page_number: int = Field(0, description="Legacy field for backward-compatibility.")
    text_snippet: str = Field(..., description="A short preview of the exact chunk the AI used.")

class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(..., description="The generated answer by the LLM for the query")
    citations: List[Citation] = Field(..., description="A list of citations to support the answer")



