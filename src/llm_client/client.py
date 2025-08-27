import json

import requests
from pydantic import BaseModel, ValidationError

# --- 1. Define the OpenAPI Specification (as a Python dictionary) ---
# This is a simplified representation of what you'd find in a real openapi.json file.
# It defines the expected request and response structures for our LLM endpoint.
openapi_spec = {
    "openapi": "3.0.0",
    "info": {"title": "Simple LLM API", "version": "1.0.0"},
    "paths": {
        "/v1/chat/completions": {
            "post": {
                "summary": "Create a chat completion",
                "requestBody": {
                    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ChatCompletionRequest"}}}
                },
                "responses": {
                    "200": {
                        "description": "Successful response",
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/ChatCompletionResponse"}}
                        },
                    }
                },
            }
        }
    },
    "components": {
        "schemas": {
            "ChatCompletionRequest": {
                "type": "object",
                "properties": {
                    "model": {"type": "string"},
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                        },
                    },
                },
                "required": ["model", "messages"],
            },
            "ChatCompletionResponse": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "object": {"type": "string"},
                    "created": {"type": "integer"},
                    "model": {"type": "string"},
                    "choices": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "message": {
                                    "type": "object",
                                    "properties": {"role": {"type": "string"}, "content": {"type": "string"}},
                                },
                            },
                        },
                    },
                },
            },
        }
    },
}

# --- 2. Define Pydantic Models for Data Validation ---
# These models are created based on the schemas in our OpenAPI spec.
# They ensure that the data we send and receive is in the correct format.


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[Message]


class ResponseChoice(BaseModel):
    index: int
    message: Message


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ResponseChoice]


# --- 3. Create the LLM Client ---


class LLMClient:
    """
    A simple client for interacting with an LLM API that follows our defined OpenAPI spec.
    """

    def __init__(self, base_url: str, api_key: str):
        """
        Initializes the client with the API's base URL and an API key.
        """
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def get_completion(self, model: str, messages: list[Message]) -> ChatCompletionResponse:
        """
        Sends a request to the LLM to get a chat completion.

        Args:
            model: The name of the model to use for the completion.
            messages: A list of message dictionaries, each with a 'role' and 'content'.

        Returns:
            A ChatCompletionResponse object with the LLM's reply.
        """
        endpoint = "/v1/chat/completions"
        url = f"{self.base_url}{endpoint}"

        try:
            # Validate the request data before sending
            request_data = ChatCompletionRequest(model=model, messages=messages)
            payload = request_data.model_dump_json()

            # Make the API call
            response = requests.post(url, headers=self.headers, data=payload)
            response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

            # Validate the response data
            response_data = response.json()
            return ChatCompletionResponse.model_validate(response_data)

        except ValidationError as e:
            print(f"Data validation error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
            raise


# --- 4. Example Usage ---

if __name__ == "__main__":
    # This is a mock server for demonstration purposes.
    # In a real scenario, you would replace this with the actual API URL.
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class MockServer(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/v1/chat/completions":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "gpt-3.5-turbo-0613",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello there! How can I assist you today?",
                            },
                        }
                    ],
                }
                self.wfile.write(json.dumps(response).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

    def run_mock_server():
        server_address = ("", 8000)
        httpd = HTTPServer(server_address, MockServer)
        httpd.serve_forever()

    # Run the mock server in a separate thread
    server_thread = threading.Thread(target=run_mock_server)
    server_thread.daemon = True
    server_thread.start()

    # --- Client in action ---
    client = LLMClient(base_url="http://localhost:8000", api_key="YOUR_API_KEY")

    messages_to_send = [
        Message(role="system", content="you are a helpful assistant"),
        Message(role="user", content="what is the meaning of life ?"),
    ]

    try:
        completion = client.get_completion(model="gpt-3.5-turbo", messages=messages_to_send)
        print("Successfully received completion:")
        print(f"  Response ID: {completion.id}")
        print(f"  Model used: {completion.model}")
        print(f"  Assistant's reply: {completion.choices[0].message.content}")

    except (ValidationError, requests.exceptions.RequestException) as e:
        print(f"An error occurred: {e}")
