"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import pandas as pd
import numpy as np

from transformers import (
    LlamaForCausalLM, LlamaTokenizer
)
from peft import PeftModel
from pymatgen.core import Structure, Element
from pymatgen.core.lattice import Lattice

MAX_LENGTH = 2048
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def parse_fn(gen_str):
    lines = [x for x in gen_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [x for x in lines[2::2]]
    coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]
    
    structure = Structure(
        lattice=Lattice.from_parameters(
            *(lengths + angles)),
        species=species,
        coords=coords, 
        coords_are_cartesian=False,
    )
    
    return structure.to(fmt="cif")

def find_similar_elements(target_element, elements, tolerance=0.1):
    similar_elements = []
    for state, radius in target_element.ionic_radii.items():
        for el in elements:
            if state in el.ionic_radii:
                radius_diff = abs(radius - el.ionic_radii[state])
                if radius_diff < tolerance and el.symbol != target_element.symbol:
                    similar_elements.append((el.symbol, state, radius_diff))
    return sorted(similar_elements, key=lambda x: x[2])

def make_swap_table(tolerance=0.1):
    elements = [Element(el) for el in Element]

    swap_table = {}

    for el in elements:
        swap_table[el.symbol] = [
            x[0] for x in find_similar_elements(el, elements, tolerance=tolerance)
        ]

    return swap_table

def get_crystal_string(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")

    # Randomly translate within the unit cell
    structure.translate_sites(
        indices=range(len(structure.sites)), vector=np.random.uniform(size=(3,))
    )

    lengths = structure.lattice.parameters[:3]
    angles = structure.lattice.parameters[3:]
    atom_ids = structure.species
    frac_coords = structure.frac_coords

    crystal_str = \
        " ".join(["{0:.1f}".format(x) for x in lengths]) + "\n" + \
        " ".join([str(int(x)) for x in angles]) + "\n" + \
        "\n".join([
            str(t) + "\n" + " ".join([
                "{0:.2f}".format(x) for x in c
            ]) for t,c in zip(atom_ids, frac_coords)
        ])

    return crystal_str

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict, 
    llama_tokenizer, 
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def prepare_model_and_tokenizer(model_name, model_path):
    llama_options = model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def llama2_model_string(model_size, chat):
        chat = "chat-" if chat else ""
        return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    model_string = llama2_model_string(model_size, is_chat)
    print(f"Using model: {model_string}")
    
    model = LlamaForCausalLM.from_pretrained(
        model_string,
        load_in_8bit=True,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    model.eval()

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )

    model = PeftModel.from_pretrained(model, model_path, device_map="auto")
    
    return model, tokenizer
        
def unconditional_sample(model, tokenizer, num_samples, batch_size, temperature, top_p, instruction_prompt):

    prompts = []
    for _ in range(num_samples):
        prompt = "Below is a description of a bulk material. "
        prompt += instruction_prompt
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        prompts.append(prompt)
 
    outputs = []
    while len(outputs) < num_samples:
        batch_prompts = prompts[len(outputs):len(outputs)+batch_size]

        batch = tokenizer(
            list(batch_prompts), 
            return_tensors="pt",
        )
        batch = {k: v.cuda() for k, v in batch.items()}

        generate_ids = model.generate(
            **batch,
            do_sample=True,
            max_new_tokens=500,
            temperature=temperature, 
            top_p=top_p, 
        )

        gen_strs = tokenizer.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )

        for gen_str, prompt in zip(gen_strs, batch_prompts):
            material_str = gen_str.replace(prompt, "")

            try:
                cif_str = parse_fn(material_str)
                _ = Structure.from_str(cif_str, fmt="cif")
            except Exception as e:
                print(e)
                continue

            outputs.append({
                "gen_str": gen_str,
                "cif": cif_str,
            })

    return outputs

# Use to prompt llm, only need to provide model name and model path
def prompt_llm(model_name : str, model_path : str, num_samples = 5, batch_size = 100, out_path = "llm_samples.csv", temperature = 0.9, top_p = 0.9, instruction_prompt = ""):

    if ".csv" not in out_path:
        i = os.environ.get("SLURM_ARRAY_TASK_ID", 0)
        out_path = os.path.join(out_path, f"samples_{i}.csv") 
    
    model, tokenizer = prepare_model_and_tokenizer(model_name, model_path)

    return unconditional_sample(model, tokenizer, num_samples, batch_size, out_path, temperature, top_p, instruction_prompt)
