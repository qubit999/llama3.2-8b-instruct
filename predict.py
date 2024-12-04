import os
import torch
from cog import BasePredictor, Input, Path
from transformers import pipeline
from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "./Llama-3.2-3B-Instruct"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",  # Set device_map explicitly
        )
        self.pad_token_id = self.pipe.tokenizer.eos_token_id  # Set pad_token_id to eos_token_id

    def predict(
        self,
        prompt: str = Input(description="Question", default="Name 3 animals with wings"),
    ) -> str:        
        messages = [
            {"role": "system", "content": "You are an AI chatbot."},
            {"role": "user", "content": prompt},
        ]
        outputs = self.pipe(
            messages,
            max_new_tokens=4096,
            pad_token_id=self.pad_token_id,
        )
        return outputs[0]["generated_text"][-1]

    def cleanup(self):
        """Cleanup after each prediction to save memory"""
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
        print(predictor.predict(prompt="Name 3 animals with wings"))
        predictor.cleanup()
        print(predictor.predict(prompt="if True: print('Hello, World')"))
        predictor.cleanup()