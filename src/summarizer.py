from transformers import BartTokenizer, BartForConditionalGeneration

# Load pretrained BART model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def summarize(text, max_length=500):
    """
    Takes input text and returns a summary.
    """
    inputs = tokenizer([text], return_tensors='pt')
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        early_stopping=False
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# Test it
if __name__ == "__main__":
    sample_text = """About 10 men armed with pistols and small machine guns
    raided a casino in Switzerland and made off into France with several 
    hundred thousand Swiss francs in the early hours of Sunday morning."""
    
    print("Input Text:")
    print(sample_text)
    print("\nGenerated Summary:")
    print(summarize(sample_text))