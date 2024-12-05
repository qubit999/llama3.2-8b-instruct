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
            device_map="auto",
        )
        self.pad_token_id = self.pipe.tokenizer.eos_token_id  # Set pad_token_id to eos_token_id

    def predict(
        self,
        prompt: str = Input(description="Question", default="Name 3 animals with wings"),
        system_prompt: str = Input(description="System prompt", default="You are an AI chatbot."),
        max_new_tokens: int = Input(description="Maximum number of new tokens to generate", default="4096"),
        min_new_tokens: int = Input(description="Minimum number of new tokens to generate", default="1"),
        temperature: float = Input(description="Sampling temperature", default="0.7"),
        top_p: float = Input(description="Top-p (nucleus) sampling", default="0.9"),
        top_k: int = Input(description="Top-k sampling", default=0),
        length_penalty: float = Input(description="Length penalty", default=1.0),
        repetition_penalty: float = Input(description="Repetition penalty", default=1.0),
        do_sample: bool = Input(description="Use sampling", default=True),
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        outputs = self.pipe(
            messages,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=self.pad_token_id,
            length_penalty=length_penalty,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
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
    run_bool = True
    if run_bool:
        predictor = Predictor()
        predictor.setup()
        print(predictor.predict(
            prompt="Name 3 animals with wings",
            system_prompt="You are an AI chatbot.",
            max_new_tokens=4096,
            min_new_tokens=1,
            temperature=0.7,
            top_p=0.9,
            top_k=0,
            length_penalty=1.0,
            repetition_penalty=1.0,
        ))
        predictor.cleanup()
        print(predictor.predict(prompt="if True: print('Hello, World')"))
        predictor.cleanup()