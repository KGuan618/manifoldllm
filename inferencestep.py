import argparse
import json
import torch
import random
import re
from pathlib import Path
from tqdm import tqdm
from unsloth import FastModel

def parse_step_length_entries(lines):
    length_entries = []
    current_hole_id = None

    for idx, line in enumerate(lines):
        hole_match = re.match(r"\s*Hole\s+(\d+)", line)
        if hole_match:
            current_hole_id = int(hole_match.group(1))
        if current_hole_id is not None:
            step_match = re.match(r"\s*Step\s+(\d+):\s+Diameter\s+\d+,\s+Length\s+([\d.]+)", line)
            if step_match:
                step_number = int(step_match.group(1))
                length_value = step_match.group(2)
                length_entries.append((idx, current_hole_id, step_number, length_value))
    return length_entries

def apply_length_masking(lines, entries_to_mask):
    lines_copy = lines.copy()
    for idx, hole_id, step_number, length_value in entries_to_mask:
        lines_copy[idx] = re.sub(rf"Length\s+{length_value}", "Length <masked>", lines_copy[idx])
    return lines_copy

def extract_lengths(text, output_format):
    if output_format == "hole_and_lengths":
        pattern = r"Hole\s+(\d+)\s+Step\s+(\d+)\s+Length\s+([\d.]+)"
        matches = re.findall(pattern, text)
        return [(int(h), int(s), v) for h, s, v in matches], bool(matches)
    else:
        matches = re.findall(r"[\d.]+", text)
        return matches, bool(matches)

def build_instruction(hole_steps, include_masked_holes, output_format, is_longshot=False):
    hole_steps = sorted(hole_steps)
    base = "You are given a hydraulic manifold design with several holes, each of which may have multiple steps.\n"
    base += "Each step consists of a diameter and a length.\n"

    if include_masked_holes:
        hole_list = '\n'.join(f"- Hole {hid}, Step {step}: 'Length' is masked" for hid, step in hole_steps)
        base += f"The following step lengths are masked:\n{hole_list}\n"
    else:
        base += "Some 'Length' fields have been masked in the text provided.\n"

    base += "Each masked 'Length' has been replaced with the token '<masked>'.\n"
    base += "Your task is to infer and restore the missing length values based on the rest of the information.\n"

    if output_format == "hole_and_lengths":
        base += "Output each result exactly in the format: 'Hole <ID> Step <Step> Length <value>'\n"
        base += "For example: Hole 1 Step 2 Length 43.\n"
    else:
        base += "Output a list of numeric values corresponding to the masked lengths, in the same order they appear masked.\n"

    base += "Do not include any explanations or extra text. Just provide the outputs."
    return base

def singleshot_generation(model, tokenizer, lines, steps_to_mask, device, include_masked_holes, output_format, example, mask_count):
    steps_to_mask = sorted(steps_to_mask, key=lambda x: (x[1], x[2]))
    masked_lines = apply_length_masking(lines, steps_to_mask)
    input_text = "\n".join(masked_lines)

    hole_steps = [(hole_id, step_number) for (_, hole_id, step_number, _) in steps_to_mask]
    expected_outputs = [f"Hole {hole_id} Step {step_number} Length {length}" for (_, hole_id, step_number, length) in steps_to_mask]

    instruction = build_instruction(hole_steps, include_masked_holes, output_format)

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    input_tokens = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **input_tokens,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_beams=3,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    if "</think>" in decoded:
        decoded = decoded.split("</think>")[-1]

    parsed, success = extract_lengths(decoded, output_format)

    if not success:
        print(f"⚠️ Failed to extract lengths for {hole_steps}")
        result = {
            "instruction": instruction,
            "input": input_text,
            "ground_truth": expected_outputs,
            "generated_outputs": [],
            "full_generated_output": decoded.strip(),
            "metadata": example.get("metadata", {}),
            "masking_level": mask_count,
            "successful_generation": False
        }
        return [result]

    id_map = {}
    if output_format == "hole_and_lengths":
        id_map = {(hid, step): val for hid, step, val in parsed}
    else:
        for (idx, (_, hole_id, step_number, _)) in enumerate(steps_to_mask):
            id_map[(hole_id, step_number)] = parsed[idx] if idx < len(parsed) else None

    current_lines = masked_lines.copy()
    generated_outputs = []
    success = True

    for line_idx, hole_id, step_number, length in steps_to_mask:
        gen = id_map.get((hole_id, step_number))
        if gen:
            current_lines[line_idx] = re.sub(r"Length\s+<masked>", f"Length {gen}", current_lines[line_idx], count=1)
            generated_outputs.append(f"Hole {hole_id} Step {step_number} Length {gen}")
        else:
            generated_outputs.append(None)
            success = False

    result = {
        "instruction": instruction,
        "input": input_text,
        "ground_truth": expected_outputs,
        "generated_outputs": generated_outputs,
        "full_generated_output": decoded.strip(),
        "metadata": example.get("metadata", {}),
        "masking_level": mask_count,
        "successful_generation": success
    }

    return [result]

def generate_and_save_masked(
    model_name,
    input_jsonl_path,
    output_jsonl,
    masking_counts,
    batch_size,
    max_seq_length,
    load_in_4bit,
    finetune,
    mode,
    include_masked_holes,
    output_format,
):
    output_path = Path(output_jsonl)

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )
    if finetune:
        model = model.merge_and_unload()
    device = model.device

    with open(input_jsonl_path, "r") as f:
        examples = [json.loads(line) for line in f]

    with open(output_path, "w", encoding="utf-8") as f_out:
        random.seed(123)

        for mask_count in masking_counts:
            print(f"\n🚀 Processing masking level: {mask_count} step lengths...\n")
            for example in tqdm(examples, desc=f"Masking {mask_count} Lengths"):
                lines = example["input"].splitlines()
                length_entries = parse_step_length_entries(lines)

                if len(length_entries) < mask_count:
                    print(f"Skipping example: not enough lengths to mask {mask_count} times.")
                    continue

                selected_entries = random.sample(length_entries, mask_count)
                selected_entries = sorted(selected_entries, key=lambda x: (x[1], x[2]))

                if mode == "longshot":
                    # Not included here, assumes progressive_masking_and_generation is defined elsewhere if needed
                    raise NotImplementedError("Longshot mode is not implemented in this script.")
                elif mode == "singleshot":
                    results = singleshot_generation(
                        model, tokenizer, lines, selected_entries, device,
                        include_masked_holes, output_format, example, mask_count
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                for result in results:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

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
    parser.add_argument("--mode", type=str, choices=["longshot", "singleshot"], default="singleshot")
    parser.add_argument("--include_masked_holes", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--output_format", type=str, choices=["lengths_only", "hole_and_lengths"], default="hole_and_lengths")

    args = parser.parse_args()

    generate_and_save_masked(
        model_name=args.model_name,
        input_jsonl_path=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        masking_counts=args.masking_counts,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        finetune=args.finetune,
        mode=args.mode,
        include_masked_holes=args.include_masked_holes,
        output_format=args.output_format,
    )
