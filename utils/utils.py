from transformers import AutoModelWithLMHead, AutoTokenizer, AutoConfig


def load_embedding_model(model):
    embedding_config = AutoConfig.from_pretrained(model)
    embedding_config.output_hidden_states = True
    embedding_model = AutoModelWithLMHead.from_pretrained(model, config=embedding_config)
    embedding_model.eval()
    embedding_model.to('cuda')
    embedding_tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=False, config=embedding_config)
    return embedding_model, embedding_tokenizer, embedding_config
