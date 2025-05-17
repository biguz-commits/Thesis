from typing_extensions import TypedDict, NotRequired

class State(TypedDict, total=False):
    input_query : Notrequired[str]
    label : int
    answer : Notrequired[str]
