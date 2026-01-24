import torch
from dataclasses import dataclass
from threading import Thread

from .preprocess import LLAMA_TEMPLATE
from .finetuner import LlamaFineTuner

def load_model(
    model_name,
    adapter = None,
    use_4bit = False,
    use_cpu = False
    ):

    if not use_cpu and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        use_cpu = True

    if use_cpu:
        use_4bit = False

    if adapter:
        model, tokenizer = LlamaFineTuner.load_adapter(
            base_model = model_name,
            adapter_path = adapter,
            use_4bit = use_4bit,
            use_cpu = use_cpu
        )
    else:
        finetuner = LlamaFineTuner(
            local_models_dir = model_name,
            use_4bit = use_4bit,
            use_cpu = use_cpu
        )
        finetuner.load_model()
        model, tokenizer = finetuner.model, finetuner.tokenizer

    return model, tokenizer

@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    stop_sequences: list = []

def generate(
    model,
    tokenizer,
    prompt,
    config = None,
    **kwargs
    ):

    if config is None:
        config = GenerationConfig()

    max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
    temperature = kwargs.get("temperature", config.temperature)
    top_p = kwargs.get("top_p", config.top_p)
    top_k = kwargs.get("top_k", config.top_k)
    repetition_penalty = kwargs.get("repetition_penalty", config.repetition_penalty)
    do_sample = kwargs.get("do_sample", config.do_sample)
    stop_sequences = kwargs.get("stop_sequences", config.stop_sequences)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
    }

    if do_sample and temperature > 0:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        })
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs[0][prompt_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if stop_sequences:
        for stop in stop_sequences:
            if stop in text:
                text = text[:text.index(stop)]

    return text.strip()

def generate_with_usage(
    model,
    tokenizer,
    prompt,
    config = None,
    **kwargs  
    ):

    if config is None:
        config = GenerationConfig()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    text = generate(model, tokenizer, prompt, config, **kwargs)

    completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    
    return text, usage

def generate_stream(
    model,
    tokenizer,
    prompt,
    config = None,
    **kwargs  
    ):
    from transformers import TextIteratorStreamer
    
    if config is None:
        config = GenerationConfig()

    max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
    temperature = kwargs.get("temperature", config.temperature)
    top_p = kwargs.get("top_p", config.top_p)
    do_sample = kwargs.get("do_sample", config.do_sample)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    if do_sample and temperature > 0:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
        })
    else:
        gen_kwargs["do_sample"] = False

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    for text in streamer:
        yield text
    
    thread.join()

def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))

class Chat:
    def __init__(self,
        model,
        tokenizer,
        messages = [],
        system_prompt = None,
        config = None,
        max_context_tokens = 4096
        ):

        self.model = model
        self.tokenizer = tokenizer
        self.messages = messages
        self.max_context_tokens = max_context_tokens

        if config is None:
            self.config = GenerationConfig()

        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
        self.system_prompt = system_prompt

    def send(self,
        user_input,
        stream = False,
        **kwargs   
        ):
        self.messages.append({'role':'user','content':user_input})

        prompt = ""
        for msg in self.messages:
            prompt += LLAMA_TEMPLATE.format(
                role = msg['role'],
                content = msg['content']
            )
        prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n'

        prompt = self._truncate_context(prompt)

        if stream:
            return self._stream_response(prompt, **kwargs)
        else:
            return self._get_response(prompt, **kwargs)
        
    def _get_response(self, prompt, **kwargs):
        response = generate(self.model, self.tokenizer, prompt, self.config, **kwargs)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def _stream_response(self, prompt, **kwargs):
        full_response = []
        for chunk in generate_stream(self.model, self.tokenizer, prompt, self.config, **kwargs):
            full_response.append(chunk)
            yield chunk
        
        self.messages.append({"role": "assistant", "content": "".join(full_response)})

    def _truncate_context(self,prompt):
        tokens = count_tokens(self.tokenizer, prompt)
        while tokens > self.max_context_tokens and len(self.messages) > 2:
            for i, msg in enumerate(self.messages):
                if msg['role'] != 'system':
                    self.messages.pop(i)
                    break
            prompt = ""
            for msg in self.messages:
                prompt += LLAMA_TEMPLATE.format(
                    role = msg['role'],
                    content = msg['content']
                )
            prompt += '<|start_header_id|>assistant<|end_header_id|>\n\n'
            tokens = count_tokens(self.tokenizer, prompt)

        return prompt
    
    def clear(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})
    
    def get_history(self):
        return self.messages.copy()
    
    def set_system_prompt(self, prompt):
        self.system_prompt = prompt
        if self.messages and self.messages[0]["role"] == "system":
            self.messages[0]["content"] = prompt
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

def interactive_mode(
    model,
    tokenizer,
    system_prompt = None,
    stream = True    
    ):
    """
    Run interactive chat in terminal.
    
    Commands:
        /clear  - Clear conversation history
        /system - Set new system prompt
        /history - Show conversation history
        /stream - Toggle stream
        /quit   - Exit
    """

    chat = Chat(
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt
    )

    print("\n" + "="*60)
    print("Interactive Mode (type 'quit', 'exit', 'q' to exit)")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("ANON: ").strip()

            if not user_input:
                continue
            
            if user_input.startswith("/"):
                cmd = user_input.lower().split()[0]
                
                if cmd in ["/quit", "/exit", "/q"]:
                    break
                elif cmd == "/clear":
                    chat.clear()
                    print("(conversation cleared)\n")
                    continue
                elif cmd == "/history":
                    for msg in chat.get_history():
                        print(f"[{msg['role']}]: {msg['content'][:100]}...")
                    print()
                    continue
                elif cmd == "/system":
                    new_prompt = user_input[7:].strip()
                    if new_prompt:
                        chat.set_system_prompt(new_prompt)
                        print(f"(system prompt updated)\n")
                    else:
                        print(f"Current: {chat.system_prompt}\n")
                    continue
                elif cmd == "/stream":
                    stream = not stream
                    print(f"(stream: {stream})\n")
                    continue
                elif cmd == "/help":
                    print("Commands:")
                    print("  /clear   - Clear conversation")
                    print("  /system  - Set/show system prompt")
                    print("  /history - Show conversation history")
                    print("  /stream  - Toggle stream")
                    print("  /quit    - Exit\n")
                    continue

            print("CHUD: ", end="", flush=True)

            if stream:
                for chunk in chat.send(user_input, stream=True):
                    print(chunk, end="", flush=True)
                print("\n")
            else:
                response = chat.send(user_input)
                print(f"{response}\n")

        except KeyboardInterrupt:
            break
