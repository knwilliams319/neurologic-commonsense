import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline, set_seed, PhrasalConstraint

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

class BaseLM(torch.nn.Module):
    def __init__(self, model:str = "gpt2", seed:int = 0, max_len:int = 15, num_returns:int = 1, num_beams:int = 5):
        super(BaseLM, self).__init__()
        if model == "gpt2" or model == "gpt2-small":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=self.tokenizer.eos_token_id)
            self.generator = pipeline('text-generation', model='gpt2')
        elif model == "gpt2-medium":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
            self.generator = pipeline('text-generation', model='gpt2-medium')
        elif model == "gpt-large":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2-large')
            self.generator = pipeline('text-generation', model='gpt2-large')
        else:
            raise ValueError(f"Model type ' {model} ' not supported. [BaseLM __init__()]")
        
        set_seed(seed)
        self.max_len = max_len
        self.beams = num_beams
        self.model_type = model
        self.num_returns = num_returns
    
    def forward(self):
        pass
    
    def decode(
        self, text:str,
        constrained:bool = False, 
        concepts:list = []
    ) -> str:
        
        self.model.to(device)
        self.model.eval()
        
        with torch.no_grad():
            if self.generator:
                if not constrained:
                    return self.generator(
                        text, max_new_tokens=self.max_len, 
                        num_return_sequences=self.num_returns,
                        num_beams=self.beams
                    )[0]["generated_text"]
                
                print("Cannot perform constrained generation with generator. Generating manually.")
            
            inputs = self.tokenizer(text, return_tensors="pt")
            
            if constrained:
                constraints = [
                    PhrasalConstraint(
                        self.tokenizer(token, add_special_tokens=False).input_ids
                    )
                    for token in concepts
                ]
                
                output = self.model.generate(
                    inputs["input_ids"],
                    constraints=constraints,
                    max_new_tokens=self.max_len,
                    num_beams=self.beams,
                    num_return_sequences=self.num_returns,
                    no_repeat_ngram_size=1,
                    remove_invalid_values=True,
                )
            else:
                output = self.model.generate(
                    inputs["input_ids"], 
                    max_new_tokens=self.max_len,
                    num_beams=self.beams
                )
                
            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            return output_text

lm = BaseLM(model="gpt2-medium", max_len=20)
print(lm.decode("What is the third planet from the sun?", concepts=["planet", "third", "sun"]))