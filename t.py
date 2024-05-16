from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("pinkmanlove/llama-13b-hf")

model = AutoModelForCausalLM.from_pretrained("pinkmanlove/llama-13b-hf")