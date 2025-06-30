import os
from dataclasses import dataclass, field
from unsloth import FastLanguageModel
from transformers import HfArgumentParser
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


@dataclass
class ExperimentArguments:
    pretrained_model_name_or_path: str = field(
        default=None,
        metadata={"help": "The model checkpoint or HuggingFace repo ID"},
    )
    data_dir: str = field(
        default=None,
        metadata={"help": "Path to your JSONL training file"},
    )
    from_foundation_model: bool = field(
        default=False,
        metadata={"help": "Set True if model is a base model, False if chat/instruct-tuned"},
    )

    def __post_init__(self):
        if self.pretrained_model_name_or_path is None or self.data_dir is None:
            raise ValueError("Specify both model and dataset path!")


def apply_qlora(model, max_seq_length):
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        max_seq_length=max_seq_length,
        random_state=42,
    )


def main(user_config, sft_config):
    # Load entire dataset for training only
    train_dataset = load_dataset("json", data_files=user_config.data_dir, split="train")

    # Load model + tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=user_config.pretrained_model_name_or_path,
        max_seq_length=sft_config.max_seq_length,
        load_in_4bit=True,
        device_map='auto',
    )

    # Format examples using chat template
    def format_and_tokenize(example):
        instruction = example["instruction"].strip()
        user_input = example["input"].strip()
        assistant_output = example["output"].strip()

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": assistant_output},
        ]

        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return tokenizer(
            chat_text,
            padding="max_length",
            truncation=True,
            max_length=sft_config.max_seq_length,
        )

    train_dataset = train_dataset.map(format_and_tokenize, batched=False, remove_columns=["instruction", "input", "output"])

    # Apply QLoRA
    model = apply_qlora(model, sft_config.max_seq_length)

    # Initialize trainer with only training data
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=None,  # No validation
    )

    # Train and report stats
    trainer_stats = trainer.train()
    print(trainer_stats)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")

    trainer.save_model(output_dir="./finetuned_model_final2134/lora_final")


if __name__ == "__main__":
    parser = HfArgumentParser((ExperimentArguments, SFTConfig))
    user_config, sft_config = parser.parse_args_into_dataclasses()
    print(user_config, sft_config)
    main(user_config, sft_config)
