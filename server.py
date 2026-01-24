import argparse
import asyncio
import time
import uuid
from dataclasses import dataclass

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from chud.infer import (
    load_model,
    generate,
    generate_with_usage,
    generate_stream,
    GenerationConfig,
)

from chud.preprocess import LLAMA_TEMPLATE

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "local-model"
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[list[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0

class CompletionRequest(BaseModel):
    model: str = "local-model"
    prompt: str | list[str]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[list[str]] = None
    echo: bool = False

class ChatMessage(BaseModel):
    role: str = "assistant"
    content: str

class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage

class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"

class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]

# =============================================================================
# Server State
# =============================================================================

@dataclass
class ServerState:
    model: any = None
    tokenizer: any = None
    model_name: str = "local-model"
    is_instruct: bool = False
    device: str = "cpu"

state = ServerState()
app = FastAPI(title="Local LLM API", version="1.0.0")

# =============================================================================
# Helpers
# =============================================================================

def generate_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"

def messages_to_dicts(messages: list[Message]) -> list[dict]:
    """Convert Pydantic Message objects to dicts."""
    return [{"role": m.role, "content": m.content} for m in messages]

async def stream_chat_response(prompt: str, config: GenerationConfig):
    """Async generator for streaming chat responses."""
    response_id = generate_id()
    created = int(time.time())
    
    for chunk in generate_stream(state.model, state.tokenizer, prompt, config):
        data = {
            "id": response_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": state.model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": chunk},
                "finish_reason": None
            }]
        }
        yield f"data: {data}\n\n"
        await asyncio.sleep(0)
    
    # Final chunk
    final = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": state.model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    }
    yield f"data: {final}\n\n"
    yield "data: [DONE]\n\n"

async def stream_completion_response(prompt: str, config: GenerationConfig):
    """Async generator for streaming completion responses."""
    response_id = generate_id()
    created = int(time.time())
    
    for chunk in generate_stream(state.model, state.tokenizer, prompt, config):
        data = {
            "id": response_id,
            "object": "text_completion.chunk",
            "created": created,
            "model": state.model_name,
            "choices": [{
                "index": 0,
                "text": chunk,
                "finish_reason": None
            }]
        }
        yield f"data: {data}\n\n"
        await asyncio.sleep(0)
    
    yield "data: [DONE]\n\n"

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "model_name": state.model_name,
        "device": state.device,
    }

@app.get("/v1/models")
def list_models():
    return ModelList(data=[
        ModelInfo(
            id=state.model_name,
            created=int(time.time()),
        )
    ])

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Format messages into prompt
    messages = messages_to_dicts(request.messages)

    prompt = ""
    for msg in messages:
        prompt += LLAMA_TEMPLATE.format(
            role = msg['role'],
            content = msg['content']
        )
    prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n'
    
    # Build config
    config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop_sequences=request.stop
    )
    
    # Streaming response
    if request.stream:
        return StreamingResponse(
            stream_chat_response(prompt, config),
            media_type="text/event-stream"
        )
    
    # Non-streaming response
    text, usage = generate_with_usage(state.model, state.tokenizer, prompt, config)
    
    return ChatCompletionResponse(
        id=generate_id(),
        created=int(time.time()),
        model=state.model_name,
        choices=[Choice(message=ChatMessage(content=text))],
        usage=Usage(**usage),
    )

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Build config
    config = GenerationConfig(
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop_sequences=request.stop,
    )
    
    # Handle single prompt or list
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt
    
    # Streaming (only for single prompt)
    if request.stream and len(prompts) == 1:
        return StreamingResponse(
            stream_completion_response(prompts[0], config),
            media_type="text/event-stream"
        )
    
    # Non-streaming
    choices = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    for i, prompt in enumerate(prompts):
        text, usage = generate_with_usage(state.model, state.tokenizer, prompt, config)
        
        if request.echo:
            text = prompt + text
        
        choices.append(CompletionChoice(index=i, text=text))
        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]
        total_usage["total_tokens"] += usage["total_tokens"]
    
    return CompletionResponse(
        id=generate_id(),
        created=int(time.time()),
        model=state.model_name,
        choices=choices,
        usage=Usage(**total_usage),
    )

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Local LLM API Server")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--adapter", help="LoRA adapter path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--use_4bit", action="store_true", help="4-bit quantization")
    
    args = parser.parse_args()
    
    # Load model using inference module
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model(
        model_name=args.model,
        adapter=args.adapter,
        use_4bit=args.use_4bit,
        use_cpu=args.cpu,
    )
    
    # Set state
    state.model = model
    state.tokenizer = tokenizer
    state.model_name = args.model
    state.device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    
    print(f"\nServer ready!")
    print(f"  Model: {state.model_name}")
    print(f"  Device: {state.device}")
    print(f"  Instruct mode: {state.is_instruct}")
    print(f"\nEndpoints:")
    print(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  POST http://{args.host}:{args.port}/v1/completions")
    print(f"  GET  http://{args.host}:{args.port}/v1/models")
    print(f"  GET  http://{args.host}:{args.port}/health")
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()