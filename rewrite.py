import torch
from spacy.lang.en import English
from transformers import T5ForConditionalGeneration, T5Tokenizer

class Rewrite():
    def __init__(self) -> None:
        self.context = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained("castorini/t5-base-canard").to(self.device).eval()
        self.tokenizer = T5Tokenizer.from_pretrained("castorini/t5-base-canard")
        self.nlp = English()

    def rewrite(self, query: str) -> str:
        """
        Function for rewriting querys for a given topic
                query: quret query 
                return a rewritten query based on the context for the topic. 
        """
        if self.context == []:
                self.context.append(query)
                return query

        self.context.append(query)
        src_text = " ||| ".join(self.context)
        src_text = " ".join([tok.text for tok in self.nlp(src_text)])
        to_device = self.tokenizer(src_text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)

        out = self.model.generate(to_device,max_length=64,num_beams=10,early_stopping=True,)

        rewrite_text = self.tokenizer.decode(out[0, 0:],clean_up_tokenization_spaces=True,skip_special_tokens=True)
        return rewrite_text

    def reset_context(self) -> None:
        """
        Resets context

        """
        self.context = []

def main():
    test_topic_questions = [
            "I would like to learn about GMO Food labeling.",
            "What are the pros and cons?",
            "And what about the cons?",
            "What are the EU rules?",
            "Tell me more about traceability tools.",
            "What is the role of Co-Extra?",
            "How is testing done for contamination?",
            "What\u0027s the difference between the European and US approaches?",
            "How does the DNA-based method work?",
            "How could Co-Extra improve it?"]

    tmp = Rewrite()
    for i in test_topic_questions:
        print(tmp.rewrite(query=i))

if __name__ =='__main__':
    main()
        
        
        