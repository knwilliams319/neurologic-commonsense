import torch
from transformers import AutoTokenizer, GPT2Model, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration, pipeline, set_seed, PhrasalConstraint

class BaseLM(torch.nn.Module):
    def __init__(self, model:str = "gpt2", seed:int = 0, max_len:int = 50, num_returns:int = 1, num_beams:int = 5):
        super(BaseLM, self).__init__()
        if model == "gpt2":
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=self.tokenizer.eos_token_id)
            self.generator = pipeline('text-generation', model='gpt2')
        elif model == "t5":
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.model = T5ForConditionalGeneration.from_pretrained("t5-small")
            self.generator = None
        else:
            raise ValueError(f"Model type ' {model} ' not supported. [BaseLM __init__()]")
        
        set_seed(seed)
        self.max_len = max_len
        self.beams = num_beams
        self.model_type = model
        self.num_returns = num_returns
    
    def decode(self, text:str, constrained:bool = False, concepts:list = [],):
        if self.generator:
            if not constrained:
                return self.generator(text, max_new_tokens=self.max_len, num_return_sequences=self.num_returns)
            
            print("Cannot performed constrained generation with generator. Generating manually.")
        
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
                num_beams=self.beams,
                num_return_sequences=self.num_returns,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
            )
        else:
            output = self.model.generate(inputs["input_ids"], max_new_tokens=self.max_len)
            
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output_text

lm = BaseLM(model="gpt2", max_len=100)
print(lm.decode("What is the third planet from the sun?", constrained=True, concepts=["planet", "third", "sun"]))