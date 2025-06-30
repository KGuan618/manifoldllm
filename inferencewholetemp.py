import argparse
import json
import random
import re
import torch
from pathlib import Path
from tqdm import tqdm
from unsloth import FastModel

def parse_full_hole_entries(lines):
    holes = []
    current = None
    for idx, line in enumerate(lines):
        hole_match = re.match(r"\s*Hole\s+(\d+)\s+\(.*?\)\s+at Position\s+(\([^)]+\))(?!.*Mounting|Through)", line)
        if hole_match:
            if current:
                holes.append(current)
            current = {
                "hole_id": int(hole_match.group(1)),
                "coord": hole_match.group(2),
                "coord_line": idx,
                "step_lines": [],
                "step_data": []
            }
        elif current and "Length" in line:
            step_match = re.search(r"Step\s+(\d+):.*?Length\s+([\d.]+)", line)
            if step_match:
                step_no = int(step_match.group(1))
                length = step_match.group(2)
                current["step_lines"].append(idx)
                current["step_data"].append({"step": step_no, "length": length})
        elif line.strip() == "":
            if current:
                holes.append(current)
                current = None
    if current:
        holes.append(current)
    return holes

def apply_full_hole_masking(lines, holes):
    masked = lines.copy()
    for h in holes:
        masked[h["coord_line"]] = re.sub(r"at Position\s+\([^)]+\)", "at Position <masked>", masked[h["coord_line"]])
        for idx in h["step_lines"]:
            masked[idx] = re.sub(r"Length\s+[\d.]+", "Length <masked>", masked[idx])
    return masked

def build_instruction(holes, output_format):
    base = "Predict the missing coordinate and step lengths for the following hole in a hydraulic manifold design.\n"
    base += "The masked hole has its 'Position' and all 'Length' values replaced with '<masked>'.\n"
    base += "You must output predictions in the format: 'Hole <ID> at Position (x.xxx, y.yyy, z.zzz)' and 'Hole <ID> Step <Step> Length <value>' for each step.\n"
    base += "Do not include any explanations. Just output the exact text as described.\n"
    if output_format == "hole_and_lengths":
        base += "\nHoles masked: " + ", ".join(str(h["hole_id"]) for h in holes) + "\n"
    return base

def run_generation(model, tokenizer, device, instruction, masked_lines, temperature):
    input_text = "\n".join(masked_lines)
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    input_tokens = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **input_tokens,
            max_new_tokens=512,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_beams=3,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "</think>" in decoded:
        decoded = decoded.split("</think>")[-1]
    return decoded.strip()

def generate_and_save_masked(
    model,
    tokenizer,
    device,
    input_jsonl_path,
    output_jsonl,
    masking_counts,
    temperature
):
    output_path = Path(output_jsonl)

    with open(input_jsonl_path, "r") as f:
        examples = [json.loads(line) for line in f]

    with open(output_path, "w", encoding="utf-8") as f_out:
        random.seed(123)
        for mask_count in masking_counts:
            print(f"\n🚀 Masking {mask_count} full holes per example at temperature {temperature}\n")
            for example in tqdm(examples, desc=f"Masking {mask_count} holes"):
                lines = example["input"].splitlines()
                holes = parse_full_hole_entries(lines)

                if len(holes) < mask_count:
                    continue

                selected = random.sample(holes, mask_count)
                masked_lines = apply_full_hole_masking(lines, selected)
                instruction = build_instruction(selected, output_format="hole_and_lengths")
                decoded_output = run_generation(model, tokenizer, device, instruction, masked_lines, temperature)

                result = {
                    "input": "\n".join(masked_lines),
                    "instruction": instruction,
                    "ground_truth": [
                        {
                            "hole_id": h["hole_id"],
                            "coord": h["coord"],
                            "step_lengths": h["step_data"]
                        } for h in selected
                    ],
                    "full_generated_output": decoded_output,
                    "masking_level": mask_count,
                    "temperature": temperature,
                    "metadata": example.get("metadata", {})
                }

                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()

    print(f"✅ Done! Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--masking_counts", type=int, nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=16000)
    parser.add_argument("--load_in_4bit", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--finetune", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--temperature", type=float, nargs="+", default=[0.7])

    args = parser.parse_args()

    # Load model once
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )
    if args.finetune:
        model = model.merge_and_unload()
    device = model.device

    for temp in args.temperature:
        suffix = f"_t{temp}".replace('.', '_')
        out_path = Path(args.output_jsonl)
        temp_output_jsonl = out_path.with_name(out_path.stem + suffix + out_path.suffix)

        print(f"\n🔥 Running for temperature {temp} -> {temp_output_jsonl}")
        generate_and_save_masked(
            model=model,
            tokenizer=tokenizer,
            device=device,
            input_jsonl_path=args.input_jsonl,
            output_jsonl=str(temp_output_jsonl),
            masking_counts=args.masking_counts,
            temperature=temp
        )
