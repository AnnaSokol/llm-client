import os
import unittest
from unittest.mock import Mock, patch

import requests
from dotenv import load_dotenv
from pydantic import ValidationError

from llm_client.client import ChatCompletionResponse, LLMClient, Message

# -- lade umgebungsvariablen ---

load_dotenv()

# --- Tests für den LLMClient ---


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        """Richtet den Test-Client vor jedem Test ein."""
        self.client = LLMClient(base_url=os.getenv("API_URL", "about:blank"), api_key=os.getenv("API_KEY", "fake-key"))

    @patch("requests.post")
    def test_get_completion_success(self, mock_post):
        """Testet einen erfolgreichen Aufruf zur Chat-Vervollständigung."""
        # Konfigurieren der simulierten Antwort
        mock_response_data = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-3.5-turbo-0613",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Dies ist eine Testantwort.",
                    },
                }
            ],
        }
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status.return_value = None  # Simuliert eine erfolgreiche Anfrage
        mock_post.return_value = mock_response

        # Aufruf der zu testenden Methode
        messages = [Message(role="user", content="Hallo")]
        completion = self.client.get_completion(model="gpt-3.5-turbo", messages=messages)

        # Überprüfungen (Assertions)
        self.assertIsInstance(completion, ChatCompletionResponse)
        self.assertEqual(completion.id, "chatcmpl-test")
        self.assertEqual(completion.choices[0].message.content, "Dies ist eine Testantwort.")
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_get_completion_http_error(self, mock_post):
        """Testet die Behandlung eines HTTP-Fehlers."""
        # Konfigurieren des Mocks, um einen HTTPError auszulösen
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("404 Not Found")
        mock_post.return_value = mock_response

        # Überprüfen, ob die richtige Ausnahme ausgelöst wird
        with self.assertRaises(requests.exceptions.RequestException):
            messages = [Message(role="user", content="Hallo")]
            self.client.get_completion(model="gpt-3.5-turbo", messages=messages)


if __name__ == "__main__":
    # Führt die Tests aus, wenn das Skript direkt gestartet wird.
    # Die zusätzlichen Argumente sind für Umgebungen wie Jupyter Notebooks nützlich.
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
