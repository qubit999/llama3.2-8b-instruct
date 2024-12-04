import os
import torch

os.getenv("TORCH_USE_CUDA_DSA", "1")

from cog import BasePredictor, Input, Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

MODEL_NAME = "./Llama-3.2-3B-Instruct"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto", trust_remote_code=True).eval()

    def predict(
        self,
        prompt: str = Input(description="Question", default="Write a hello world program in Python."),
    ) -> str:
        """Run a single prediction on the model"""
        messages = [
            {"role": "system", "content": "You are Qwen. You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer(text, return_tensors="pt", padding=True).to(self.model.device)

        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=4096,
            do_sample=True,
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def predict_code(
        self,
        prompt: str = Input(description="Code", default="def hello_world():"),
    ) -> str:
        """Run a single prediction on the model"""
        model_inputs = self.tokenizer([prompt], return_tensors="pt", padding=True).to(self.model.device)
        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=4096,
            do_sample=True,
        )
        generated_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return output_text

    def cleanup(self):
        """Cleanup after each prediction to save memory"""
        if self.model is not None:
            self.model.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
        # Additional cleanup if necessary

if __name__ == "__main__":
    run_bool = False
    if run_bool:
        predictor = Predictor()
        predictor.setup()
        print(predictor.predict(prompt="Write a hello world program in Python"))
        predictor.cleanup()
        print(predictor.predict_code(prompt="if True: print('Hello, World')"))
        predictor.cleanup()