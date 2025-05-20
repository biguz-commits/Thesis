# run with python -m src.thesis.app

import asyncio
import textwrap
import os
import sys
from uuid import uuid4
import gradio as gr
from langgraph.checkpoint.memory import InMemorySaver

# Assicuriamoci che i percorsi di import siano corretti
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Importa la costruzione del grafo
from llm.graph.build import create_graph

def make_user_friendly(text: str) -> str:
    """Formatta il testo per una visualizzazione più leggibile."""
    formatted = textwrap.dedent(text).strip()
    return formatted.replace("**", "").replace("\\n", "\n")

def run_app():
    # Thread ID globale per mantenere la conversazione
    THREAD_ID = str(uuid4())
    # Inizializziamo il memory saver e il grafo
    memory = InMemorySaver()
    graph = create_graph(checkpointer=memory)

    def process_question(question):
        """Versione semplificata che processa una singola domanda"""
        if not question:
            return "Per favore inserisci una domanda."
            
        # Prepara lo stato iniziale
        initial_state = {
            "input_query": question,
            "thread_id": THREAD_ID,
        }
        
        try:
            # Esegui il grafo in modo sincrono con asyncio.run
            final_state = asyncio.run(graph.ainvoke(
                initial_state, 
                config={"configurable": {"thread_id": THREAD_ID}}
            ))
            
            # Estrai la risposta
            if "answer" in final_state:
                response = make_user_friendly(final_state["answer"])
                
                # Per debugging, aggiungi info sul percorso
                if "label" in final_state:
                    node_type = "Raccomandazione" if final_state["label"] == 1 else "Risposta Generale"
                    response += f"\n\n_Classificazione: {node_type}_"
                
                return response
            else:
                return "Nessuna risposta generata dall'agente."
                
        except Exception as e:
            return f"Errore durante l'elaborazione: {str(e)}"

    # Interfaccia Gradio semplificata
    demo = gr.Interface(
        fn=process_question,
        inputs=gr.Textbox(
            placeholder="Fai una domanda o chiedi una raccomandazione...",
            label="La tua domanda"
        ),
        outputs=gr.Textbox(label="Risposta dell'assistente"),
        title="🧠 AI Product Assistant",
        description="Chiedi informazioni o raccomandazioni sui prodotti. L'assistente utilizza BERT per classificare la tua domanda e indirizzarla al modello più appropriato.",
        examples=[
            "Qual è la differenza tra machine learning e deep learning?",
            "Puoi consigliarmi un laptop per il machine learning?",
            "Quali sono le caratteristiche principali di TensorFlow?",
            "Raccomandami un buon monitor per data science."
        ],
        theme=gr.themes.Soft(primary_hue="blue")
    )
    
    return demo

# Avvio dell'applicazione
if __name__ == "__main__":
    demo = run_app()
    # Utilizziamo share=True per risolvere il problema di localhost
    demo.launch(share=True)