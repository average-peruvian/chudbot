import argparse

def response(model_path, lora_adapter, prompt):
    from chud.finetuner import LlamaFineTuner, generate_text

    model, tokenizer = LlamaFineTuner.load_adapter(model_path, lora_adapter)
    output = generate_text(model, tokenizer, prompt)
    
    print(output)

def main():
    parser = argparse.ArgumentParser(description='Pull a response from a trained chudbot.')
    parser.add_argument('model_path', help='Path to the original model.')
    parser.add_argument('lora_adapter', help='Path to the LoRA adapter.')
    parser.add_argument('prompt', help='Your prompt.')

    args = parser.parse_args()
    response(args.model_path,args.lora_adapter,args.prompt)
