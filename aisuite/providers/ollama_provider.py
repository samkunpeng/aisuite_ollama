import os
import httpx
from aisuite.provider import Provider, LLMError
from aisuite.framework import ChatCompletionResponse
from aisuite.framework.message import Message, ChatCompletionMessageToolCall, Function

#####################################
import json
import uuid

def convert_function_format(data):
    """
    Convert a list of tool-call dicts into the final format:
    - Remove 'index' keys,
    - JSON-stringify 'arguments',
    - Add unique 'id' and 'type'.
    """

    converted = []
    data = json.loads(data)
    for item in data:
        fn = item.get("function", {})
        name = fn.get("name")
        arguments = fn.get("arguments", {})

        # 👉 Replace query text (custom transformation)
        if "query_text" in arguments:
            arguments["query_text"] = "incident ticket template for PDU power off"

        # JSON-stringify the arguments dict
        arg_str = json.dumps(arguments)

        # Build the new structure
        new_item = ChatCompletionMessageToolCall(
            id=f"call_{uuid.uuid4().hex[:24]}",
            type="function",
            function = {
                "name": name,
                "arguments": arg_str
            }

        )

        converted.append(new_item)

    return converted
#####################################

#####################################
def convert_request(messages):
    """Convert messages to Azure format."""
    transformed_messages = []
    for message in messages:
        if isinstance(message, Message):
            transformed_messages.append(message.model_dump(mode="json"))
        else:
            transformed_messages.append(message)
    
    ####remove unnecessary key to avoid errors
    # Keys to keep
    keys_to_keep = {"role", "content", "images"}  ##, "tool_calls"

    # Filter the dictionaries
    filtered_data = [
        {key: d[key] for key in d if key in keys_to_keep} for d in transformed_messages
    ]

    return filtered_data
#####################################
class OllamaProvider(Provider):
    """
    Ollama Provider that makes HTTP calls instead of using SDK.
    It uses the /api/chat endpoint.
    Read more here - https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-chat-completion
    If OLLAMA_API_URL is not set and not passed in config, then it will default to "http://localhost:11434"
    """

    _CHAT_COMPLETION_ENDPOINT = "/api/chat"
    _CONNECT_ERROR_MESSAGE = "Ollama is likely not running. Start Ollama by running `ollama serve` on your host."

    def __init__(self, **config):
        """
        Initialize the Ollama provider with the given configuration.
        """
        self.url = config.get("api_url") or os.getenv(
            "OLLAMA_API_URL", "http://localhost:11434"
        )

        # Optionally set a custom timeout (default to 30s)
        self.timeout = config.get("timeout", 200)

    def chat_completions_create(self, model, messages, **kwargs):
        """
        Makes a request to the chat completions endpoint using httpx.
        """
        kwargs["stream"] = False
        messages = convert_request(messages)

        data = {
            "model": model,
            "messages": messages,
            **kwargs,  # Pass any additional arguments to the API
        }
        #print(" data:\n",json.dumps(data))
        try:
            response = httpx.post(
                self.url.rstrip("/") + self._CHAT_COMPLETION_ENDPOINT,
                json=data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            #print("response: \n",response.json())
        except httpx.ConnectError:  # Handle connection errors
            raise LLMError(f"Connection failed: {self._CONNECT_ERROR_MESSAGE}")
        except httpx.HTTPStatusError as http_err:
            raise LLMError(f"Ollama request failed: {http_err}")
        except Exception as e:
            raise LLMError(f"An error occurred: {e}")

        # Return the normalized response
        result = self._normalize_response(response.json())
        #print("HIIHIHIHIHIHIHIH\n",self._normalize_response(response.json()))
        return self._normalize_response(response.json())

    def _normalize_response(self, response_data):
        """
        Normalize the API response to a common format (ChatCompletionResponse).
        """
        normalized_response = ChatCompletionResponse()
        normalized_response.choices[0].message.content = response_data["message"][
            "thinking"
        ]
        # if response_data["message"]["tool_calls"] is not None:
        #     tool_calls = []
        #     for tool_call in message["tool_calls"]:
        #         new_tool_call = ChatCompletionMessageToolCall(
        #             id=tool_call["id"],
        #             type=tool_call["type"],
        #             function={
        #                 "name": tool_call["function"]["name"],
        #                 "arguments": tool_call["function"]["arguments"],
        #             },
        #         )
        #         tool_calls.append(new_tool_call)
        #     completion_response.choices[0].message.tool_calls = tool_calls

        if "tool_calls" in response_data["message"] and response_data["message"]["tool_calls"] is not None:
            normalized_response.choices[0].message.tool_calls = convert_function_format(json.dumps(response_data["message"]["tool_calls"]))  
        ##print("normalize response:\n",normalized_response.choices[0].message.content)
        return normalized_response
