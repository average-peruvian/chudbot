import argparse

from chud.preprocess import format_prompt
from chud.infer import (
    load_model, 
    GenerationConfig,
    generate,
    generate_stream,
    interactive_mode
)

def main():
    parser = argparse.ArgumentParser(
        description="Run inference with finetuned model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive chat
    python inference.py --model 3b --adapter ./output
    
    # Single prompt
    python inference.py --model 3b --prompt "Hello!"
    
    # CPU inference
    python inference.py --model 3b --adapter ./output --cpu
    
    # With system prompt
    python inference.py --model 3b-instruct --system "You are a pirate."
    
    # Disable streaming
    python inference.py --model 3b --no-stream
"""
    )
    
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--adapter", help="LoRA adapter path")
    parser.add_argument("--prompt", help="Single prompt (non-interactive)")
    parser.add_argument("--system", help="System prompt")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--use_4bit", action="store_true", help="4-bit quantization")
    parser.add_argument("--max_tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(
        model_name=args.model,
        adapter=args.adapter,
        use_4bit=args.use_4bit,
        use_cpu=args.cpu,
    )
    
    config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    
    if args.prompt:
        # Single prompt mode
        prompt = format_prompt(
            args.prompt,
            system_prompt=args.system,
        ) + '<|start_header_id|>assistant<|end_header_id|>\n\n'
        
        if args.no_stream:
            response = generate(model, tokenizer, prompt, config)
            print(f"CHUD: {response}")
        else:
            print("CHUD: ", end="", flush=True)
            for chunk in generate_stream(model, tokenizer, prompt, config):
                print(chunk, end="", flush=True)
            print()
    else:
        # Interactive mode
        interactive_mode(
            model=model,
            tokenizer=tokenizer,
            system_prompt=args.system,
            stream=not args.no_stream,
        )

if __name__ == "__main__":
    main()