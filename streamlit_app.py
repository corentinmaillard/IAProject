import streamlit as st
import requests
import os
import re
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b7
from PIL import Image
from dotenv import load_dotenv
from langchain.agents import tool, AgentExecutor, ZeroShotAgent
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import torch
from torchvision import transforms
from PIL import Image
from langchain.memory import ConversationBufferWindowMemory  # Import memory module

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the LLM instance
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Ensure you're using the correct model
    temperature=0.7,
    openai_api_key=openai_api_key
)

# Define the recycling information tool
@tool
def get_recycling_info(query: str) -> str:
    """Extracts waste item and region from query, fetches data, returns recycling information and recycling location."""
    # Use LLM to extract waste item and region
    prompt_template = PromptTemplate(
        input_variables=["input_text"],
        template=(
            "Identify and extract the type of waste and the region (Bruxelles or Wallonie) mentioned in the following text for recycling purposes. "
            "If the region is not specified, default to 'Bruxelles'. "
            "Provide your answer in the format: 'Waste Item: <waste_item>; Region: <region>'.\n\nInput: {input_text}"
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)
    extraction = chain.run(input_text=query)
    
    # Parse the extraction
    match = re.search(r'Waste Item:\s*(.*?); Region:\s*(.*)', extraction, re.IGNORECASE)
    if match:
        waste_item = match.group(1).strip()
        region = match.group(2).strip()
    else:
        return "Sorry, I couldn't extract the waste item and region from your query."
    
    # Fetch data based on region
    if region.lower() in ["bruxelles", "brussels"]:
        url = "https://data.bep.be/api/records/1.0/search/?dataset=guide-de-tri&q=&rows=-1"
    elif region.lower() in ["wallonie", "wallonia"]:
        # Replace with the actual API URL when available
        url = "https://fake-api-walonia.com/recycling-data"
    else:
        return f"Region '{region}' not recognized. Please specify 'Bruxelles' or 'Wallonie'."
    
    response = requests.get(url)
    if response.status_code != 200:
        return "Failed to fetch data from the API."
    
    data = response.json()
    # Search for the waste item in the data
    results = []
    for record in data.get("records", []):
        fields = record.get("fields", {})
        title = fields.get("dechet", "")
        description = fields.get("destination", "")
        recycling_point = fields.get("infoparc", "") or fields.get("infocollecte", "")
        if waste_item.lower() in title.lower():
            if recycling_point:
                results.append(f"{title}: Où recycler: {recycling_point}")
            else:
                results.append(f"{title}: {description}. Où recycler: Non spécifié.")
    
    if results:
        return "\n".join(results)
    else:
        return f"No recycling information found for '{waste_item}' in {region}."

# Define the waste recognition tool
@tool
def waste_recognition(image_path: str) -> str:
    """Recognizes the type of waste from an image using a trained CNN model."""
    try:
        # Load the saved model checkpoint
        checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))
        
        # Define your model architecture
        output_shape = 6  # Adjust this to match the number of classes in your model
        model = efficientnet_b7()
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=2560, out_features=output_shape, bias=True)
        )
        
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # Move the model to the appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Define the image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust to match your model's expected input size
            transforms.ToTensor(),
            # Add any other transformations used during training
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension
        image = image.to(device)  # Move image to the same device as the model
        
        # Perform inference
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)
        
        # Map the predicted class index to the waste item label
        # Define your class names in the same order as during training
        class_names = ['carton', 'verre', 'métal', 'papier', 'plastique', 'déchet']  # Replace with your actual class names
        
        waste_item = class_names[predicted.item()]
        
        return waste_item
    except Exception as e:
        return f"An error occurred during waste recognition: {e}"

# Initialize the tools
tools = [get_recycling_info, waste_recognition]

# Define the agent prompt using ZeroShotAgent
agent_prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix = (
        "You are an assistant that must rely on tools to provide accurate and detailed answers. "
        "Whenever a user asks a question or provides an image, your first step is to determine which tool to use. "
        "Use the tool `waste_recognition` when the input is an image. "
        "When you use `waste_recognition`, interpret the result to determine the type of waste, "
        "and then use the tool `get_recycling_info` to retrieve information about how to recycle it. "
        "When you use the tool `get_recycling_info`, ensure all input is in French and contains only plain words (no special characters). "
        "Use the tool `get_recycling_info` directly when the input is text. "
        "Do not answer questions directly without consulting the tools. "
        "When you provide your Final Answer, ensure that:"
        "- All waste items with the same recycling process and location are grouped together. "
        "- If multiple groups exist, list them explicitly, numbered or in bullet points. "
        "- For each group, provide a clear summary of the items, followed by where to recycle them. "
        "Your response must be structured as a numbered list, detailed, and easy to understand."
    ),
    suffix="{input}\n\n{agent_scratchpad}",
    input_variables=["input", "agent_scratchpad"]
)

# Create the LLMChain
llm_chain = LLMChain(llm=llm, prompt=agent_prompt)

# Create the agent
agent = ZeroShotAgent(
    llm_chain=llm_chain,
    allowed_tools=[tool.name for tool in tools]
)

# Add Memory Management
memory = ConversationBufferWindowMemory(k=1)  # Keep only the last interaction

# Create the AgentExecutor with memory
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True )

# Streamlit app title
st.title("Recycling Guide with Image Recognition and API Integration")

# Main function
def generate_response(input_text=None, image_file=None):
    if openai_api_key and openai_api_key.startswith("sk-"):
        if image_file is not None:
            # Save the uploaded image to a temporary file
            temp_image_path = "temp_image.jpg"
            with open(temp_image_path, "wb") as f:
                f.write(image_file.read())
            # un the agent with the image
            response = agent_executor.run(temp_image_path)
            st.success(response)

            os.remove(temp_image_path)
        elif input_text:
            # Run the agent with the text input
            response = agent_executor.run(input_text)
            st.success(response)
            # **Reset the agent's memory to prevent context buildup**
            agent_executor.memory.clear()
        else:
            st.error("Please provide a valid input.")
    else:
        st.error("Please provide a valid OpenAI API key.")

# User input
option = st.selectbox(
    "Choose input type:",
    ("Text Input", "Image Upload")
)

if option == "Text Input":
    with st.form("text_form"):
        text = st.text_area(
            "Enter your query:",
            "Comment recycler une bouteille à Bruxelles?"
        )
        submitted = st.form_submit_button("Submit")
        if submitted:
            generate_response(input_text=text)
elif option == "Image Upload":
    image_file = st.file_uploader("Upload an image of the waste item:", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        if st.button("Submit"):
            generate_response(image_file=image_file)
