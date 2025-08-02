from typing_extensions import TypedDict, NotRequired

class State(TypedDict, total=False):
    input_query : NotRequired[str]
    label : int
    answer : NotRequired[str]
