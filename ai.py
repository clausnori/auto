import aiohttp
import asyncio
import re
from typing import List

class AIChatClient:
    """
    Асинхронный класс для общения с AI через Gemini API (без Gradio и озвучки)
    """

    def __init__(self, api_key: str, system_prompt: str = ""):
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.history = []
        self.url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.headers = {"Content-Type": "application/json"}

    def extract_text_from_tags(self, text: str) -> List[str]:
        pattern = r'%(.*?)%'
        return re.findall(pattern, text, re.DOTALL)

    def remove_tags_from_text(self, text: str) -> str:
        pattern = r'%.*?%'
        matches = re.findall(pattern, text, re.DOTALL)
        return ' '.join(matches).strip()

    def get_text_without_tags(self, text: str) -> str:
        pattern = r'%.*?%'
        return re.sub(pattern, '', text, flags=re.DOTALL).strip()

    async def get_text_response(self, query: str) -> str:
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": self.system_prompt + "\n" + query if self.system_prompt else query}
                    ]
                }
            ]
        }

        url_with_key = f"{self.url}?key={self.api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url_with_key, headers=self.headers, json=data) as response:
                    response.raise_for_status()
                    content = await response.json()
                    text = content['candidates'][0]['content']['parts'][0]['text']
                    print(f"[AI Response]: {text}")
                    return text
        except Exception as e:
            print(f"Ошибка при получении ответа от Gemini: {e}")
            return "Извините, произошла ошибка при обработке вашего запроса."
            
    def extract_text_from_bracket_tags(self, text: str) -> List[str]:
	    """
	    Извлекает содержимое внутри %[ ... ]%
	    """
	    pattern = r'%\[(.*?)\]%'
	    return re.findall(pattern, text, re.DOTALL)
    
    async def get_response(self, query: str) -> str:
        response = await self.get_text_response(query)
        self.history.append((query, response))
        return response

    def clear_history(self):
        self.history = []

    def set_system_prompt(self, prompt: str):
        self.system_prompt = prompt

    def set_api_key(self, api_key: str):
        self.api_key = api_key