import os
import torch
from cog import BasePredictor, ConcatenateIterator, Input, Path
from transformers import pipeline
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

load_dotenv()

MODEL_NAME = "./Llama-3.2-3B-Instruct"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"
DEFAULT_SYSTEM_PROMPT = "You are an AI chatbot."

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipe = pipeline("text-generation", model=MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    def predict(
        self,
        prompt: str = Input(description="Question", default="Name 3 animals with wings"),
        system_prompt: str = Input(description="System prompt", default=DEFAULT_SYSTEM_PROMPT),
        max_new_tokens: int = Input(description="Maximum number of new tokens to generate", default=4096),
        min_new_tokens: int = Input(description="Minimum number of new tokens to generate", default=1),
        temperature: float = Input(description="Sampling temperature", default=0.7),
        top_p: float = Input(description="Top-p (nucleus) sampling", default=0.9),
        top_k: int = Input(description="Top-k sampling", default=0),
        length_penalty: float = Input(description="Length penalty", default=1.0),
        repetition_penalty: float = Input(description="Repetition penalty", default=1.0),
        do_sample: bool = Input(description="Use sampling", default=True),
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        if isinstance(max_new_tokens, int):
            max_new_tokens = max_new_tokens
        else:
            max_new_tokens = max_new_tokens.default
        if isinstance(min_new_tokens, int):
            min_new_tokens = min_new_tokens
        else:
            min_new_tokens = min_new_tokens.default
        if isinstance(temperature, float):
            temperature = temperature
        else:
            temperature = temperature.default
        if isinstance(top_p, float):
            top_p = top_p
        else:
            top_p = top_p.default
        if isinstance(top_k, int):
            top_k = top_k
        else:
            top_k = top_k.default
        if isinstance(length_penalty, float):
            length_penalty = length_penalty
        else:
            length_penalty = length_penalty.default
        if isinstance(repetition_penalty, float):
            repetition_penalty = repetition_penalty
        else:
            repetition_penalty = repetition_penalty.default
        if isinstance(do_sample, bool):
            do_sample = do_sample
        else:
            do_sample = do_sample.default

        outputs = self.pipe(
            messages,
            max_length=max_new_tokens,
            min_length=min_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.pipe.tokenizer.eos_token_id,  # Explicitly set pad_token_id
        )
        return outputs[0]["generated_text"][-1]

    def cleanup(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

if __name__ == "__main__":
    run = False
    if run:
        predictor = Predictor()
        predictor.setup()
        print(predictor.predict())
        predictor.cleanup()