from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("my_fine_tuned_model")

prompt_text = "Once upon a time"
encoded_prompt = tokenizer.encode(
    prompt_text, add_special_tokens=False, return_tensors="pt"
)

# Set pad token
tokenizer.pad_token = tokenizer.eos_token

# Now create your attention mask
attention_mask = encoded_prompt.ne(tokenizer.pad_token_id).long()

output_sequences = model.generate(
    input_ids=encoded_prompt,
    attention_mask=attention_mask,
    max_length=100,
    temperature=1.0,
    top_k=0,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,  # use EOS token for padding
)

if len(output_sequences.shape) > 2:
    output_sequences.squeeze_()

generated_sequence = output_sequences[0].tolist()
text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

print(text)
