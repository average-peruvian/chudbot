"""
Finetuning LLaMa using LoRA for memory efficiency :3
"""

import torch
from dataclasses import dataclass
from pathlib import Path
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig

MODELS = {
    "1b": "llama-3.2-1b",
    "1b-instruct": "llama-3.2-1b-instruct",
    "3b": "llama-3.2-3b",
    "3b-instruct": "llama-3.2-3b-instruct",
    "8b": "llama-3.1-8b",
    "8b-instruct": "llama-3.1-8b-instruct"
}

def find_local_model(shorthand, base_dir = './models'):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        return None
    
    model_path = base_dir / MODELS[shorthand.lower()]
    if model_path.exists():
        return str(model_path).lower()

@dataclass
class LoRAParams:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules = [
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        'gate_proj', 'up_proj', 'down_proj'
    ]

@dataclass
class TrainingParams:
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_strategy: str = 'epoch'

class LlamaFineTuner:
    def __init__(self,
        output_dir = './llama-finetuned',
        local_models_dir = None,
        use_4bit = True,
        use_8bit = False,
        use_cpu = False
        ):

        self.model_name = local_models_dir or ''
        self.output_dir = Path(output_dir)
        self.use_cpu = use_cpu

        if use_cpu:
            self.use_4bit = False
            self.use_8bit = False
            print("CPU mode enabled - quantization disabled")
        else:
            self.use_4bit = use_4bit
            self.use_8bit = use_8bit

        self.model = None
        self.tokenizer = None
        self.trainer = None

    def _get_quantization_config(self):
        if self.use_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        elif self.use_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        return None
    
    def load_model(self, device_map = None):
        print(f'Loading {self.model_name}...')

        if device_map is None:
            device_map = "cpu" if self.use_cpu else "auto"

        bnb_config = self._get_quantization_config()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True    
        )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'

        if self.use_cpu:
            # CPU: use float32 for best compatibility, or bfloat16 if supported
            torch_dtype = torch.float32
        elif bnb_config:
            torch_dtype = None  # Let quantization handle it
        else:
            torch_dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True
        )

        if self.use_4bit or self.use_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        print("Model loaded successfully!")
        return self
    
    def setup_lora(self, params = None):
        if params is None:
            params = LoRAParams()

        lora_config = LoraConfig(
            r=params.r,
            lora_alpha=params.lora_alpha,
            lora_dropout=params.lora_dropout,
            target_modules=params.target_modules,
            bias='none',
            task_type='CAUSAL_LM'
        )

        self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()
        return self
    
    def train(self,
        dataset,
        params = None,
        resume_from_checkpoint = False
        ):
        if params is None:
            params = TrainingParams()

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.use_cpu:
            use_bf16 = False
            use_fp16 = False
            optim = "adamw_torch"
            dataloader_pin_memory = False
            gradient_checkpointing = False  # Can cause issues on CPU
        else:
            use_bf16 = torch.cuda.is_bf16_supported()
            use_fp16 = not use_bf16 and torch.cuda.is_available()
            optim = "paged_adamw_8bit" if (self.use_4bit or self.use_8bit) else "adamw_torch"
            dataloader_pin_memory = True
            gradient_checkpointing = True

        training_args = SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=params.epochs,
            per_device_train_batch_size=params.batch_size,
            gradient_accumulation_steps=params.gradient_accumulation_steps,
            learning_rate=params.learning_rate,
            weight_decay=params.weight_decay,
            warmup_ratio=params.warmup_ratio,
            lr_scheduler_type='cosine',
            logging_steps=params.logging_steps,
            save_strategy=params.save_strategy,
            bf16=use_bf16,
            fp16=use_fp16,
            optim=optim,
            gradient_checkpointing=gradient_checkpointing,
            report_to='none',
            dataloader_pin_memory=dataloader_pin_memory,
            max_length=params.max_seq_length,
        )

        self.trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            args=training_args
        )
        
        print('Starting training...')
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        self.trainer.save_model(str(self.output_dir))
        self.tokenizer.save_pretrained(str(self.output_dir))
        print(f'Model saved to {self.output_dir}')
        return self
    
    def save_adapter(self, path: str = None):
        save_path = Path(path) if path else self.output_dir / 'adapter'
        self.model.save_pretrained(str(save_path))
        print(f'Adapter saved to {save_path}')
    
    def merge_and_save(self, output_dir = None):
        merged_dir = Path(output_dir) if output_dir else self.output_dir / 'merged'
        merged_dir.mkdir(parents=True, exist_ok=True)
        
        print('Merging LoRA weights...')
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(str(merged_dir))
        self.tokenizer.save_pretrained(str(merged_dir))
        print(f'Merged model saved to {merged_dir}')
        
        return merged_dir
    
    @classmethod
    def load_adapter(
        cls,
        base_model,
        adapter_path,
        use_4bit = True,
        use_cpu = False
        ):
        finetuner = cls(local_models_dir=base_model, use_4bit=use_4bit, use_cpu=use_cpu)
        finetuner.load_model()
        
        finetuner.model = PeftModel.from_pretrained(
            finetuner.model,
            adapter_path
        )
        finetuner.model.eval()
        
        return finetuner.model, finetuner.tokenizer