import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm.client import get_model
from langchain_core.messages import HumanMessage, AIMessage


def test_model_response():
    print("Inizializzazione del modello...")
    model = get_model()

    print("Invio richiesta...")
    test_message = [HumanMessage(content="Chi è Daniele Biganzoli?")]
    response = model.invoke(test_message)

    assert isinstance(response, AIMessage), "La risposta non è un AIMessage!"
    assert response.content.strip() != "", "La risposta è vuota!"

    print("Test passato ✅")
    print("\nRisposta del modello:")
    print(response.content)


if __name__ == "__main__":
    test_model_response()
