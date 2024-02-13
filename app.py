from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
from llama_cpp import Llama
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
model = AutoModelForCausalLM.from_pretrained(
  "stabilityai/stablelm-3b-4e1t",
  trust_remote_code=True,
  torch_dtype="auto"
)
model.cpu()

llm = Llama(
        model_path="./stablelm-zephyr-3b.Q4_K_M.gguf",  # Download the model file first
        n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
        n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
        n_gpu_layers=35         # The number of layers to offload to GPU, if you have GPU acceleration available
        )

with st.sidebar:
    Your_Name = st.text_input("Enter your name")
    "[Indulge in the Gen AI world](https://github.com/BalaInnovationHub/)"



st.title(Your_Name+": Your private AI agent!")
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a your personal chatbot. How can I help you?"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Tell me about India?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)


    with st.chat_message("assistant"):
        #st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        output = llm(
            "<|user|>\n{prompt}<|endoftext|>\n<|assistant|>", # Prompt
            max_tokens=512,  # Generate up to 512 tokens
            stop=["</s>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
            echo=True        # Whether to echo the prompt
            )
        # inputs = tokenizer("\n", return_tensors="pt").to("cpu")
        # tokens = model.generate(
        #     **inputs,
        #     max_new_tokens=100,
        #     temperature=0.75,
        #     top_p=0.95,
        #     do_sample=True,
        #     )
        response=output
        #response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
