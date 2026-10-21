"""OpenAI Responses and Gemini adapters sharing one request's tool execution state."""

import asyncio
import base64
import inspect
import json
import logging

from google.genai import types


log = logging.getLogger(__name__)
OPENAI_MODEL = "gpt-6-luna"
GEMINI_MODEL = "gemini-3.8-flash"
MAX_TOOL_ROUNDS = 5
PRIMARY_TIMEOUT_SECONDS = 35

TOOL_DEFINITIONS = {
    "knowledge_base": {
        "description": "Search the company's knowledge base for facts, prices and procedures.",
        "properties": {"message": {"type": "string"}},
    },
    "human_handoff": {
        "description": "Hand the current conversation to a human when live chat is available.",
        "properties": {"summary": {"type": "string"}, "client_name": {"type": "string"}},
    },
    "not_resolved": {
        "description": "Mark the current conversation as unresolved.",
        "properties": {},
    },
}
MUTATING_TOOLS = {"human_handoff", "not_resolved"}


class ModelResponseError(RuntimeError):
    pass


class ToolRunner:
    def __init__(self, handlers):
        self.handlers = handlers
        self.results = {}
        self.history = []

    def definitions(self):
        return [
            {
                "type": "function",
                "name": name,
                "description": TOOL_DEFINITIONS[name]["description"],
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": TOOL_DEFINITIONS[name]["properties"],
                    "required": list(TOOL_DEFINITIONS[name]["properties"]),
                    "additionalProperties": False,
                },
            }
            for name in self.handlers
        ]

    async def run(self, name, arguments):
        if name not in self.handlers:
            return {"error": "Unknown tool"}
        if not isinstance(arguments, dict):
            return {"error": "Tool arguments must be an object"}
        properties = TOOL_DEFINITIONS[name]["properties"]
        if any(key not in properties or not isinstance(value, str) for key, value in arguments.items()):
            return {"error": "Invalid tool arguments"}
        try:
            bound = inspect.signature(self.handlers[name]).bind(**arguments)
            bound.apply_defaults()
        except TypeError:
            return {"error": "Missing or invalid tool arguments"}

        # Side effects must never be repeated when switching providers, even if
        # the second model supplies a different summary or client name.
        key = (name, "" if name in MUTATING_TOOLS else json.dumps(bound.arguments, sort_keys=True))
        if key in self.results:
            return self.results[key]
        interrupted = {"error": "Tool execution was interrupted; do not retry this action."}
        if name in MUTATING_TOOLS:
            self.results[key] = interrupted
            record = {"name": name, "arguments": dict(bound.arguments), "result": interrupted}
            self.history.append(record)
        else:
            record = {"name": name, "arguments": dict(bound.arguments)}
        try:
            result = await self.handlers[name](**bound.arguments)
        except Exception:
            log.exception("Chat tool failed: %s", name)
            result = {"error": "Tool failed. Do not claim this action succeeded."}
        self.results[key] = result
        record["result"] = result
        if name not in MUTATING_TOOLS:
            self.history.append(record)
        return result


def openai_content(user_input, attachments):
    content = []
    for downloaded in attachments:
        attachment = downloaded.attachment
        data_url = f"data:{attachment.type};base64,{base64.b64encode(downloaded.content).decode('ascii')}"
        content.append({"type": "input_text", "text": f"Załącznik: {attachment.name}"})
        if attachment.type == "application/pdf":
            content.append({"type": "input_file", "filename": attachment.name, "file_data": data_url})
        else:
            content.append({"type": "input_image", "image_url": data_url})
    content.append({"type": "input_text", "text": user_input})
    return content


async def openai_reply(http_client, api_key, system_prompt, user_input, runner, attachments):
    if not api_key:
        raise ModelResponseError("OPENAI_API_KEY is not configured")
    inputs = [{"role": "user", "content": openai_content(user_input, attachments)}]
    for round_index in range(MAX_TOOL_ROUNDS + 1):
        response = await http_client.post(
            "https://api.openai.com/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": OPENAI_MODEL,
                "instructions": system_prompt,
                "input": inputs,
                "tools": runner.definitions(),
                "parallel_tool_calls": False,
                "reasoning": {"effort": "low"},
                "max_output_tokens": 8192,
                "store": False,
                "include": ["reasoning.encrypted_content"],
            },
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "completed":
            raise ModelResponseError("OpenAI response did not complete")
        output = data.get("output", [])
        calls = [item for item in output if item.get("type") == "function_call"]
        if not calls:
            text = "".join(
                part.get("text", part.get("refusal", ""))
                for item in output if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") in {"output_text", "refusal"}
            ).strip()
            if not text:
                raise ModelResponseError("OpenAI returned an empty response")
            return text
        if round_index == MAX_TOOL_ROUNDS:
            raise ModelResponseError("OpenAI exceeded the tool round limit")
        # Keep reasoning items (including encrypted reasoning) and call IDs.
        inputs.extend(output)
        for call in calls:
            try:
                arguments = json.loads(call["arguments"])
            except (ValueError, KeyError, TypeError):
                arguments = None
            result = await runner.run(call["name"], arguments)
            inputs.append({
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps(result, ensure_ascii=False, default=str),
            })
    raise ModelResponseError("OpenAI exceeded the tool round limit")


async def gemini_reply(client, system_prompt, user_input, runner, attachment_inputs):
    if client is None:
        raise ModelResponseError("Gemini client is not initialized")
    if runner.history:
        user_input += "\n\n<previous_tool_results>\n" + json.dumps(
            runner.history, ensure_ascii=False, default=str,
        ) + "\n</previous_tool_results>"
        system_prompt += "\nThe previous_tool_results section contains tool results from this request. Do not repeat completed actions."
    chat = client.aio.chats.create(
        model=GEMINI_MODEL,
        config=types.GenerateContentConfig(
            tools=list(runner.handlers.values()),
            system_instruction=system_prompt,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    message = [*attachment_inputs, types.Part.from_text(text=user_input)]
    for round_index in range(MAX_TOOL_ROUNDS + 1):
        response = await chat.send_message(message)
        candidates = getattr(response, "candidates", None) or []
        if candidates and candidates[0].finish_reason == types.FinishReason.MAX_TOKENS:
            raise ModelResponseError("Gemini response was truncated")
        calls = response.function_calls
        if not calls:
            text = (response.text or "").strip()
            if not text:
                raise ModelResponseError("Gemini returned an empty response")
            return text
        if round_index == MAX_TOOL_ROUNDS:
            raise ModelResponseError("Gemini exceeded the tool round limit")
        message = []
        for call in calls:
            result = await runner.run(call.name, call.args or {})
            message.append(types.Part(function_response=types.FunctionResponse(
                id=call.id,
                name=call.name,
                response={"result": result},
            )))
    raise ModelResponseError("Gemini exceeded the tool round limit")


async def generate_reply(http_client, api_key, gemini_client, system_prompt, user_input,
                         handlers, attachments, prepare_gemini_inputs):
    runner = ToolRunner(handlers)
    try:
        async with asyncio.timeout(PRIMARY_TIMEOUT_SECONDS):
            return await openai_reply(http_client, api_key, system_prompt, user_input, runner, attachments)
    except Exception as error:
        # Avoid logging request bodies or credentials from provider errors.
        log.warning("OpenAI unavailable (%s); falling back to %s", type(error).__name__, GEMINI_MODEL)
    attachment_inputs = await prepare_gemini_inputs(attachments)
    return await gemini_reply(gemini_client, system_prompt, user_input, runner, attachment_inputs)
