import gradio as gr
import os
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import spaces

# Check if we're running in a Hugging Face Space with GPU constraints
IS_SPACES_ZERO = os.environ.get("SPACES_ZERO_GPU", "0") == "1"
IS_SPACE = os.environ.get("SPACE_ID", None) is not None

# Get Hugging Face token from environment variables
HF_TOKEN = os.environ.get('HF_TOKEN')

# Determine device (use GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"
LOW_MEMORY = os.getenv("LOW_MEMORY", "0") == "1"

print(f"Using device: {device}")
print(f"Low memory mode: {LOW_MEMORY}")

# Model configuration
load_in_4bit = True  # Use 4-bit quantization if memory is constrained

# Load model and tokenizer with device mapping
model_name = "nafisneehal/chandler_bot"
model = AutoPeftModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=load_in_4bit
)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define prompt structure (update as needed for your model)
alpaca_prompt = "{instruction} {input_text} {output}"


@spaces.GPU  # Use GPU provided by Hugging Face Spaces if available
def generate_response(user_input, chat_history):
    instruction = "Chat with me like Chandler talks."
    input_text = user_input  # Treats user input as the input

    # Format the input using the prompt template
    formatted_input = alpaca_prompt.format(
        instruction=instruction, input_text=input_text, output="")

    # Prepare inputs for model inference on the correct device
    inputs = tokenizer([formatted_input], return_tensors="pt").to(device)

    # Generate response on GPU or CPU as appropriate
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)

    # Decode response and remove the instruction part
    bot_reply = tokenizer.decode(
        outputs[0], skip_special_tokens=True).replace(instruction, "").strip()

    # Update chat history with user and bot interactions
    chat_history.append(("User", user_input))
    chat_history.append(("Bot", bot_reply))

    return chat_history, ""  # Returns updated chat history and clears input


# Set up Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Chandler-Like Chatbot on GPU")

    chat_history = gr.Chatbot(label="Chat History", elem_id="chatbox")
    user_input = gr.Textbox(
        placeholder="Type your message here...", label="Your Message")

    # Connect submit actions to generate response function
    user_input.submit(generate_response, [user_input, chat_history], [
                      chat_history, user_input])
    submit_btn = gr.Button("Send")
    submit_btn.click(generate_response, [user_input, chat_history], [
                     chat_history, user_input])

# Custom CSS to align chat labels on the left
demo.css = """
#chatbox .bot, #chatbox .user {
    text-align: left;
}
"""

demo.launch()  # Enables a public link
