# Accelerating computational catalysis with GenAI and foundational models

<figure>
  <img src="flow.png" alt="Image">
  <figcaption> System overview: The LLM generates a string representation of a bulk material, which is converted into a 3D atomic structure along with an adsorbate. This model is then processed by an EGNN for property prediction, and the results are stored in a database. </figcaption>
</figure>

## 🛠 Installation
TODO: add install.sh
Run the following command to install all dependencies. 
```
source install.sh
```

## Example usage
```
python src/pkg/main.py --model_name=7b --model_path=path/to/model/checkpoint --adsorbate=N2 --out_path=path/to/output/folder
```
## Required arguments:
|   Argument | Explanation |
| ---------: | :----------------------- |
|  `--model_name`  | Specify which llama 2 model to use, for example "7b" or "70b-chat" |
| `--model_path`  | Specify path to fine-tuned weights|
| `--adsorbate` | Which adsorbate to use in string format, for example "H2" |
| `--out_path` | Output path, all databases, CIF, and trajectory files will be stored under this directory |

## Optional arguments:
|   Argument | Explanation |
| ---------: | :----------------------- |
| `--surface_site_sampling_mode` | Choose how adsorption sites are chosen. "random", "heuristic" or "random_site_heuristic_placement"  |
| `--ml_model_checkpoint` | Specify the ML model checkpoint to use for property prediction  |
|   `--num_samples` | Number of samples/catalysts the llm will generate  |
|  `--batch_size` |  Batch size of LLM  |
| `--samples_file` | use a csv file containing samples/catalysts instead of generating them  |
|   `--temperature`  | temperature of LLM |
|  `--top_p`  | top_p of LLM |
|  `--instruction_prompt`  | Instruction promp for LLM material generation, for example "The chemical formula is PtRu" |