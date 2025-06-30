import argparse
import json
import torch
import random
import re
from pathlib import Path
from tqdm import tqdm
from unsloth import FastModel
def extract_reasoning(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def parse_coordinate_entries(lines):
    coord_entries = []
    for idx, line in enumerate(lines):
        if line.strip().startswith("Hole") and "Mounting" not in line and "Through" not in line:
            try:
                hole_id = int(line.strip().split()[1])
                if "at Position" in line:
                    start = line.index("at Position") + len("at Position")
                    end = line.index(")", start) + 1
                    coord_str = line[start:end].strip()
                    coord_entries.append((idx, hole_id, coord_str))
            except Exception:
                continue
    return coord_entries

def apply_masking(lines, entries_to_mask):
    lines_copy = lines.copy()
    for idx, hole_id, coord_str in entries_to_mask:
        lines_copy[idx] = lines_copy[idx].replace(f"at Position {coord_str}", "at Position <masked>")
    return lines_copy

def extract_coordinate(text, output_format):
    if output_format == "hole_and_coordinates":
        coord_pattern = r"Hole\s+\d+\s+at Position\s+(\([-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+\))"
    else:
        coord_pattern = r"(\([-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+,\s*[-+]?\d*\.?\d+\))"
    match = re.search(coord_pattern, text)
    return (match.group(1), True) if match else (None, False)

def build_instruction(hole_ids, include_masked_holes, output_format, is_longshot=False, chain_of_thought=False):
    hole_ids = sorted(hole_ids)
    base = "Predict the missing coordinates for this hydraulic manifold design.\n"

    if include_masked_holes:
        hole_list = ', '.join(map(str, hole_ids))
        base += f"The 'Position' for Holes {hole_list} have been masked in the text provided.\n"
    else:
        base += "The 'Position' fields have been masked in the text provided.\n"

    base += "Each missing 'Position' has been replaced with the token '<masked>'.\n"

    if chain_of_thought:
        base += (
            "Think step by step to deduce the missing positions. For each masked hole, provide a brief reasoning "
            "of 3–5 lines only, enclosed in <think> ... </think>. After your reasoning, output the coordinates in the required format.\n"
        )

    base += "Identify and output the exact coordinates"
    if include_masked_holes:
        base += " for each Hole."

    if output_format == "hole_and_coordinates":
        base += "\nYour output must be precisely formatted as: 'Hole <ID> at Position (x.xxx, y.yyy, z.zzz)'."
    elif output_format == "coordinates_only":
        base += "\nYour output must be a list of coordinates in the form: (x.xxx, y.yyy, z.zzz) — one for each masked hole, ordered."

    base += "\nDo not include any additional text outside of reasoning and final formatted answers."
    return base

def progressive_masking_and_generation(model, tokenizer, lines, holes_to_mask, device, include_masked_holes, output_format, example, mask_count):
    masked_lines = apply_masking(lines, holes_to_mask)
    current_lines = masked_lines.copy()
    results = []
    initial_mask_count = mask_count

    for hole_idx, (line_idx, hole_id, coord_str) in enumerate(holes_to_mask):
        input_text = "\n".join(current_lines)
        instruction = build_instruction([hole_id], include_masked_holes, output_format, is_longshot=True, chain_of_thought=True)

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        input_tokens = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **input_tokens,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_beams=3,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        if "</think>" in decoded:
            decoded = decoded.split("</think>")[-1]

        coordinate, success = extract_coordinate(decoded, output_format)
        results.append({
            "instruction": instruction,
            "input": input_text,
            "ground_truth": f"Hole {hole_id} at Position {coord_str}",
            "generated_output": f"Hole {hole_id} at Position {coordinate}" if success else None,
            "full_generated_output": decoded.strip(),
            "successful_generation": success,
            "masking_level": initial_mask_count,
            "metadata": example.get("metadata", {})
        })

        if success:
            current_lines[line_idx] = current_lines[line_idx].replace("at Position <masked>", f"at Position {coordinate}")

    return results

def singleshot_generation(model, tokenizer, lines, holes_to_mask, device, include_masked_holes, output_format, example, mask_count):
    holes_to_mask = sorted(holes_to_mask, key=lambda x: x[1])
    masked_lines = apply_masking(lines, holes_to_mask)
    input_text = "\n".join(masked_lines)

    hole_ids = [hole_id for (_, hole_id, _) in holes_to_mask]
    expected_outputs = [f"Hole {hole_id} at Position {coord}" for (_, hole_id, coord) in holes_to_mask]
    instruction = build_instruction(hole_ids, include_masked_holes, output_format, chain_of_thought=True)

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    input_tokens = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **input_tokens,
            max_new_tokens=4000,  # Increased token limit
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_beams=3,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    reasoning = extract_reasoning(decoded)

    if "</think>" in decoded:
        decoded_after_thinking = decoded.split("</think>")[-1]
    else:
        decoded_after_thinking = decoded

    if output_format == "hole_and_coordinates":
        coord_pattern = r"Hole\s+(\d+)\s+at Position\s+(\([-+]?[0-9]*\.?[0-9]+,\s*[-+]?[0-9]*\.?[0-9]+,\s*[-+]?[0-9]*\.?[0-9]+\))"
        matches = re.findall(coord_pattern, decoded_after_thinking)
        id_to_coord = {int(hole_id): coord for hole_id, coord in matches}
    else:
        coord_pattern = r"\([-+]?[0-9]*\.?[0-9]+,\s*[-+]?[0-9]*\.?[0-9]+,\s*[-+]?[0-9]*\.?[0-9]+\)"
        matches = re.findall(coord_pattern, decoded_after_thinking)
        id_to_coord = {hole_id: coord for hole_id, coord in zip(hole_ids, matches)}

    success = True
    current_lines = masked_lines.copy()
    generated_outputs = []

    for line_idx, hole_id, coord_str in holes_to_mask:
        if hole_id in id_to_coord:
            coordinate = id_to_coord[hole_id]
            current_lines[line_idx] = current_lines[line_idx].replace("at Position <masked>", f"at Position {coordinate}", 1)
            generated_outputs.append(f"Hole {hole_id} at Position {coordinate}")
        else:
            print(f"⚠️ Missing prediction for Hole {hole_id}")
            generated_outputs.append(None)
            success = False

    return [{
        "instruction": instruction,
        "input": input_text,
        "ground_truth": expected_outputs,
        "generated_outputs": generated_outputs,
        "full_generated_output": decoded.strip(),
        "reasoning": reasoning,
        "metadata": example.get("metadata", {}),
        "masking_level": mask_count,
        "successful_generation": success
    }]
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
            print(f"\n🚀 Processing masking level: {mask_count} holes...\n")
            for example in tqdm(examples, desc=f"Masking {mask_count} Holes"):
                lines = example["input"].splitlines()
                coord_entries = parse_coordinate_entries(lines)

                if len(coord_entries) < mask_count:
                    print(f"Skipping example: not enough holes to mask {mask_count} times.")
                    continue

                selected_entries = sorted(random.sample(coord_entries, mask_count), key=lambda x: x[1])

                if mode == "longshot":
                    results = progressive_masking_and_generation(
                        model, tokenizer, lines, selected_entries, device,
                        include_masked_holes, output_format, example, mask_count
                    )
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
    parser.add_argument("--mode", type=str, choices=["longshot", "singleshot"], default="longshot")
    parser.add_argument("--include_masked_holes", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--output_format", type=str, choices=["coordinates_only", "hole_and_coordinates"], default="hole_and_coordinates")

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
