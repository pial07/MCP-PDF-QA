from tiktoken import get_encoding

class OpenAITokenizerWrapper:
    def __init__(self, model_name: str = "cl100k_base"):
        self.tokenizer = get_encoding(model_name)

    def tokenize(self, text: str):
        return self.tokenizer.encode(text)

    def __call__(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab