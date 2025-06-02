from typing_extensions import TypedDict, NotRequired
from typing import List, Dict, Any, Optional

class State(TypedDict, total=False):
    input_query : NotRequired[str]
    label : int
    answer : NotRequired[str]
    conversation_history: List[Dict[str, str]]  
    session_id: str  
    context_summary: Optional[str]
