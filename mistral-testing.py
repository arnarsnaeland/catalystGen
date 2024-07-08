from unsloth import FastLanguageModel
from transformers import pipeline
import torch
import sys

model_to_test = sys.argv[1]
input_string = ""
if len(sys.argv) > 2:
    input_string = sys.argv[2]

max_seq_length = 25600 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_to_test,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)
messages = []
#if there is a string to test the model with then use it
if input_string != "":
    messages = [
        {"role": "user", "content": '''You are a catalysis expert that provides good catalyst options for queries from a user,
         each response should consist of five options, given in a valid JSON format like shown in the example between <<<>>>. Do not include any explanation in the response, only the JSON. Do not include <<<>>> in the response.
        The user query is denoted by "Query : "

         <<<{"catalysts": [{"type": "Single-atom catalyst", "composition": "Cu-Ni", "atomic-formula_with_ratios": "Cu1Ni1", "space_group": "Fm-3m", "facet": "111"},]}>>> ''' "Query: " + input_string}
    ]

else :
    messages = [
        {"role": "user", "content": "What is the effect of Fe doping on cobalt surfaces?"},
    ]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

pipe = pipeline("conversational", model=model, tokenizer=tokenizer, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
result = pipe(messages)
print(result.messages[-1]['content'])


outputs = model.generate(inputs, pad_token_id=tokenizer.eos_token_id, max_new_tokens = 512, use_cache = True)
print(tokenizer.batch_decode(outputs))