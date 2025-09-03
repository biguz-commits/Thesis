import requests
import os
from dotenv import load_dotenv
from typing import Optional, Dict, List, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()


class OpenRouterModel(BaseChatModel):
    model_name: str = "nvidia/llama-3.1-nemotron-nano-8b-v1:free"
    temperature: float = 0.6
    max_tokens: int = 2048
    api_key: Optional[str] = None
    api_endpoint: str = "https://openrouter.ai/api/v1/chat/completions"
    headers: Optional[Dict[str, str]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = self.api_key or os.getenv("OPENROUTER_API_KEY")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    @property
    def _llm_type(self) -> str:
        return "openrouter-chat"

    def _format_messages(self, messages):
        formatted = []
        has_user_message = False

        for m in messages:
            if isinstance(m, HumanMessage):
                formatted.append({"role": "user", "content": m.content})
                has_user_message = True
            elif isinstance(m, AIMessage):
                formatted.append({"role": "assistant", "content": m.content})
            elif isinstance(m, SystemMessage):
                formatted.append({"role": "system", "content": m.content})
            else:
                raise ValueError(f"Tipo di messaggio non supportato: {type(m)}")

        if not has_user_message:
            formatted.append({"role": "user", "content": "I need product recommendations."})

        return formatted

    def invoke(self, messages, **kwargs):
        formatted_messages = self._format_messages(messages)

        system_exists = any(msg.get("role") == "system" for msg in formatted_messages)
        if not system_exists:
            formatted_messages.insert(0, {"role": "system", "content": "detailed thinking on"})

        data = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        response = requests.post(self.api_endpoint, headers=self.headers, json=data)
        if response.status_code == 200:
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            return AIMessage(content=content)
        else:
            raise Exception(f"Errore API: {response.status_code} - {response.text}")

    def _generate(self, messages, stop=None, **kwargs):
        ai_message = self.invoke(messages, **kwargs)
        return {"messages": [ai_message]}


def get_model(
    model_name: str = "mistralai/mistral-7b-instruct",
    temperature=0.6,
    max_tokens=2048
):
    model = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return model
