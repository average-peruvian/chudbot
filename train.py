import argparse
from pathlib import Path

from chud.preprocess import DataProcessor, load_posts
from chud.utils import project_config
from chud.finetuner import (
    LlamaFineTuner,
    LoRAParams,
    TrainingParams,
    find_local_model
)

def main():
    parser = argparse.ArgumentParser(
        description="Finetune Llama models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", "-c",
        default="config/train_config.yaml",
        help="Path to YAML config file"
    )
    args = parser.parse_args()

    config = project_config(args.config,'train')

    local_dir = find_local_model(config['model']['name'],config['model']['local_models_dir'])

    print("=== LOADING DATA ===")
    posts = load_posts(config["data"]["input_file"])

    print("\n=== PROCESSING DATA ===")
    processor = DataProcessor(
        min_length=config["data"]["min_length"],
        max_length=config["data"]["max_length"],
        deduplicate=config["data"]["deduplicate"]
    )

    dataset = processor.create_dataset(
        posts,
        mode=config["data"]["mode"],
        model_name=config['model']['name'],
        system_prompt=config["model"].get("system_prompt")
    )
    print(f"Dataset size: {len(dataset)} examples")

    print("=== LOADING MODEL ===")
    
    finetuner = LlamaFineTuner(
        output_dir=config["training"]["output_dir"],
        local_models_dir=local_dir,
        use_4bit=config["model"]["use_4bit"],
        use_8bit=config["model"]["use_8bit"]
    )
    finetuner.load_model(device_map=config['model']['device'])
    
    lora_params = LoRAParams(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["alpha"],
        lora_dropout=config["lora"]["dropout"]
    )
    finetuner.setup_lora(lora_params)

    print("\n=== TRAINING ===")
    
    training_params = TrainingParams(
        epochs=config["training"]["epochs"],
        batch_size=config["training"]["batch_size"],
        learning_rate=config["training"]["learning_rate"],
        max_seq_length=config["training"]["max_seq_length"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_ratio=config["training"]["warmup_ratio"],
        weight_decay=config["training"]["weight_decay"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
    )
    
    finetuner.train(dataset, training_params)

    if config["training"]["merge_weights"]:
        print("\n=== MERGING WEIGHTS ===")
        finetuner.merge_and_save()

    print("\n=== COMPLETE ===")
    print(f"Model: {config['model']['name']}")
    print(f"Mode: {config['data']['mode']}")
    print(f"LoRA adapter saved to: {config['training']['output_dir']}")
    if config["training"]["merge_weights"]:
        print(f"Merged model saved to: {config['training']['output_dir']}/merged")

if __name__ == "__main__":
    main()
