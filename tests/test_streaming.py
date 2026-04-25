import pytest
import asyncio
from claude_to_openai_forwarder.translators.streaming import StreamingTranslator
import json
from typing import AsyncIterator

@pytest.mark.asyncio
async def test_translate_stream():
    # Mock OpenAI stream
    async def mock_openai_stream() -> AsyncIterator[bytes]:
        data = [
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" World!"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n'
        ]
        for chunk in data:
            yield chunk

    # Test translate_stream
    stream = StreamingTranslator.translate_stream(mock_openai_stream())
    events = []
    async for event in stream:
        events.append(event)

    # Verify events
    assert len(events) == 7
    for event in events:
        lines = event.split('\n')
        for line in lines:
            if line.startswith('data: '):
                data = json.loads(line.replace('data: ', ''))
                if data['type'] == 'message_start':
                    assert data['message']['id'] is not None
                elif data['type'] == 'content_block_start':
                    assert data['content_block']['type'] == 'text'
                elif data['type'] == 'content_block_delta':
                    assert data['delta']['text'] in ['Hello', ' World!']
                elif data['type'] == 'content_block_stop':
                    assert data['index'] is not None
                elif data['type'] == 'message_delta':
                    assert data['delta']['stop_reason'] == 'end_turn'
                elif data['type'] == 'message_stop':
                    assert True

@pytest.mark.asyncio
async def test_translate_stream_with_tool_calls():
    # Mock OpenAI stream with tool calls
    async def mock_openai_stream_with_tool_calls() -> AsyncIterator[bytes]:
        data = [
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"name":"my_tool","arguments":"{\"arg\":\"value\"}"}}]},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DONE]\n\n'
        ]
        for chunk in data:
            yield chunk

    # Test translate_stream with tool calls
    stream = StreamingTranslator.translate_stream(mock_openai_stream_with_tool_calls())
    events = []
    async for event in stream:
        events.append(event)

    # Verify events
    assert len(events) == 3
    for event in events:
        lines = event.split('\n')
        for line in lines:
            if line.startswith('data: '):
                data = json.loads(line.replace('data: ', ''))
                if data['type'] == 'message_start':
                    assert data['message']['id'] is not None
                elif data['type'] == 'content_block_start':
                    assert data['content_block']['type'] == 'tool_use'
                elif data['type'] == 'content_block_delta':
                    assert data['delta']['partial_json'] == '{\"arg\":\"value\"}'
                elif data['type'] == 'content_block_stop':
                    assert data['index'] is not None
                elif data['type'] == 'message_delta':
                    assert data['delta']['stop_reason'] == 'tool_use'
                elif data['type'] == 'message_stop':
                    assert True

@pytest.mark.asyncio
async def test_stream_simulation():
    # Simulate OpenAI stream
    async def mock_openai_stream() -> AsyncIterator[bytes]:
        data = [
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"This"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{"content":" test"},"finish_reason":null}]}\n\n',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":123,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: [DONE]\n\n'
        ]
        for chunk in data:
            await asyncio.sleep(0.1)
            yield chunk

    # Test translate_stream
    stream = StreamingTranslator.translate_stream(mock_openai_stream())
    events = []
    async for event in stream:
        events.append(event)

    # Verify events
    assert len(events) == 9
    response_content = ''
    for event in events:
        lines = event.split('\n')
        for line in lines:
            if line.startswith('data: '):
                data = json.loads(line.replace('data: ', ''))
                if data['type'] == 'content_block_delta':
                    response_content += data['delta']['text']

    assert response_content == 'This is a test'