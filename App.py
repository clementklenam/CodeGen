import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "nvidia/llama-3.1-nemotron-70b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

st.title("Code Generation with NVIDIA LLaMA 3.1")

# Input box for the code prompt
user_prompt = st.text_area("Enter your prompt for code generation:")

if st.button("Generate Code"):
    if user_prompt:
        inputs = tokenizer(user_prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150)
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Display the generated code
        st.code(generated_code)
    else:
        st.warning("Please enter a prompt.")
