import os
import asyncio
import aiohttp
import discord
from discord.ext import commands
import ollama
from dotenv import load_dotenv
import logging
import json
import base64
from io import BytesIO

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Your Discord bot token and Ollama API URL
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3.1')
COMFYUI_API_URL = os.getenv('COMFYUI_API_URL', 'http://127.0.0.1:8188')

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Initialize Ollama client
ollama_client = ollama.Client(host=OLLAMA_API_URL)

class ConversationManager:
    def __init__(self):
        """
        Initializes a new ConversationManager instance.

        This method is called when a new ConversationManager object is created.
        It sets up the initial state of the object, including an empty dictionary to store conversations.
        """
        self.conversations = {}

    def get_conversation(self, user_id):
        """
        Retrieves the conversation history for a given user ID.

        Parameters:
            user_id (str): The ID of the user to retrieve the conversation history for.

        Returns:
            list: The conversation history for the given user ID.
        """
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        return self.conversations[user_id]

    def add_message(self, user_id, role, content):
        """
        Adds a message to the conversation history of a user.

        Parameters:
            user_id (str): The ID of the user to add the message for.
            role (str): The role of the message (e.g. 'user' or 'assistant').
            content (str): The content of the message.

        Returns:
            None
        """
        conversation = self.get_conversation(user_id)
        conversation.append({"role": role, "content": content})
        # Keep only the last 10 messages to avoid token limit issues
        self.conversations[user_id] = conversation[-10:]

conversation_manager = ConversationManager()


async def generate_response(prompt, with_memory=True, user_id=None):
    """
    Generates a response to a given prompt using the Ollama API.

    Parameters:
        prompt (str): The input prompt to generate a response for.
        with_memory (bool): Whether to use conversation history to generate the response. Defaults to True.
        user_id (str): The ID of the user to retrieve conversation history for. Defaults to None.

    Returns:
        str: The generated response to the prompt.
    """
    try:
        if with_memory and user_id:
            conversation = conversation_manager.get_conversation(user_id)
            conversation.append({"role": "user", "content": prompt})
            full_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        else:
            full_prompt = prompt

        # Create a new Ollama client for each request
        ollama_client = ollama.AsyncClient(host=OLLAMA_API_URL)
        
        response = await ollama_client.generate(model=OLLAMA_MODEL, prompt=full_prompt)
        
        if with_memory and user_id:
            conversation_manager.add_message(user_id, "assistant", response['response'])
        
        return response['response']
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response at this time."

@bot.command(name='models')
async def list_models(ctx):
    """
    Sends a GET request to the Ollama API to retrieve a list of available models.
    """
    async with ctx.typing():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{OLLAMA_API_URL}/api/tags") as response:
                    if response.status == 200:
                        models = await response.json()
                        model_list = "\n".join([model['name'] for model in models['models']])
                        await ctx.send(f"Available models:\n```\n{model_list}\n```")
                    else:
                        await ctx.send(f'Error: {response.status} - Failed to fetch model list')
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            await ctx.send(f'An error occurred while fetching the model list. Please try again later.')

async def generate_image(prompt):
    """
    Generates an image based on the given prompt.

    Args:
        prompt (str): The prompt for the image generation.

    Returns:
        Optional[BytesIO]: The generated image as a BytesIO object, or None if an error occurred.
    """
    workflow = {
        "3": {
            "inputs": {
                "seed": 359819880975166,
                "steps": 15,
                "cfg": 1,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1,
                "model": [
                    "4",
                    0
                ],
                "positive": [
                    "6",
                    0
                ],
                "negative": [
                    "7",
                    0
                ],
                "latent_image": [
                    "5",
                    0
                ]
            },
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {
                "ckpt_name": "flux1-dev-fp8.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": 512,
                "height": 512,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": prompt,
                "clip": [
                    "4",
                    1
                ]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": "text, watermark",
                "clip": [
                    "4",
                    1
                ]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": [
                    "3",
                    0
                ],
                "vae": [
                    "4",
                    2
                ]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                "filename_prefix": "ComfyUI",
                "images": [
                    "8",
                    0
                ]
            },
            "class_type": "SaveImage"
        }
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{COMFYUI_API_URL}/prompt", json={"prompt": workflow}) as response:
                if response.status == 200:
                    data = await response.json()
                    prompt_id = data['prompt_id']
                    
                    while True:
                        async with session.get(f"{COMFYUI_API_URL}/history/{prompt_id}") as history_response:
                            if history_response.status == 200:
                                history_data = await history_response.json()
                                if prompt_id in history_data:
                                    output_data = history_data[prompt_id]['outputs']
                                    logger.debug(f"Output data: {json.dumps(output_data, indent=2)}")
                                    if output_data and '9' in output_data:
                                        image_data = output_data['9']['images'][0]
                                        logger.debug(f"Image data type: {type(image_data)}")
                                        logger.debug(f"Image data content: {json.dumps(image_data, indent=2)}")
                                        
                                        if isinstance(image_data, str):
                                            # If image_data is a base64 string
                                            image_data = base64.b64decode(image_data.split(",", 1)[1])
                                        elif isinstance(image_data, dict):
                                            # If image_data is a dictionary
                                            if 'filename' in image_data and 'subfolder' in image_data:
                                                # The image was saved on the server, we need to retrieve it
                                                filename = image_data['filename']
                                                subfolder = image_data['subfolder']
                                                image_url = f"{COMFYUI_API_URL}/view?filename={filename}&subfolder={subfolder}"
                                                async with session.get(image_url) as img_response:
                                                    if img_response.status == 200:
                                                        image_data = await img_response.read()
                                                    else:
                                                        logger.error(f"Failed to retrieve image from server: {img_response.status}")
                                                        return None
                                            else:
                                                logger.error(f"Unexpected image data format: {image_data}")
                                                return None
                                        else:
                                            logger.error(f"Unexpected image data format: {type(image_data)}")
                                            return None
                                        return BytesIO(image_data)
                        await asyncio.sleep(1)
                else:
                    logger.error(f"Error generating image: {response.status}")
                    return None
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return None

@bot.event
async def on_ready():
        """
        Event handler that is called when the bot has successfully connected to the Discord server.

        This function logs a message indicating that the bot has connected to Discord using the `logger.info` method.
        The message includes the username of the bot user.

        Parameters:
            None

        Returns:
            None
        """
        logger.info(f'{bot.user} has connected to Discord!')

@bot.command(name='chat')
async def chat(ctx, *, message):
    """
    Handles the 'chat' command.
    
    Parameters:
        ctx: The context of the command invocation.
        message: The message to be processed.
    
    Returns:
        None
    """
    async with ctx.typing():
        response = await generate_response(message, with_memory=True, user_id=ctx.author.id)
        await ctx.send(response)

@bot.command(name='message')
async def message(ctx, *, message):
    """
    Command to send a message to the bot for a one-time response without memory.
    
    Parameters:
        ctx (Context): The context in which the message was sent.
        message (str): The message sent to the bot.
    
    Returns:
        None
    """
    async with ctx.typing():
        response = await generate_response(message, with_memory=False)
        await ctx.send(response)

@bot.command(name='clear')
async def clear_conversation(ctx):
    """
    Clears the conversation history of the user who invoked this command.

    Parameters:
        ctx (discord.ext.commands.Context): The context of the command invocation.

    Returns:
        None
    """
    conversation_manager.conversations[ctx.author.id] = []
    await ctx.send("Your conversation history has been cleared.")

@bot.command(name='photo')
async def photo(ctx, *, prompt):
    """
    Sends a generated image to the current channel if the `prompt` parameter is not empty. The `prompt` parameter is a string that serves as input for the image generation model. The function takes in a `ctx` object, which is an instance of the `discord.ext.commands.Context` class, and returns nothing (`None`). If the `prompt` is empty, the function sends a message to the current channel indicating that it couldn't generate an image at this time.

    Parameters:
    - `ctx` (discord.ext.commands.Context): The context object for the command invocation.
    - `prompt` (str): The prompt for the image generation model.

    Returns:
    - None: Nothing.

    Raises:
    - `discord.ext.commands.CommandInvokeError`: If there's an error in the command execution.

    Notes:
    - The function generates an image using the `generate_image` function from the `generate_image` module.
    - The generated image is sent to the current channel using the `ctx.send` method.
    - The generated image is saved in a file named "generated_image.png".
    """
    async with ctx.typing():
        image_data = await generate_image(prompt)
        if image_data:
            await ctx.send(file=discord.File(fp=image_data, filename="generated_image.png"))
        else:
            await ctx.send("Sorry, I couldn't generate an image at this time.")

@bot.event
async def on_message(message):
    """
    Event triggered when a message is received from the Discord server.
    
    This function checks if the message is from the bot itself and returns immediately if so.
    If the message mentions the bot, it generates a response using the generate_response function
    with memory enabled and the user ID of the message author, then sends the response back to the channel.
    
    The function also processes any commands that may be present in the message.
    
    Parameters:
        message (discord.Message): The message received from the Discord server.
    
    Returns:
        None
    """
    if message.author == bot.user:
        return
    if bot.user.mentioned_in(message):
        async with message.channel.typing():
            response = await generate_response(message.content, with_memory=True, user_id=message.author.id)
            await message.reply(response)
    await bot.process_commands(message)

@bot.event
async def on_command_error(ctx, error):
    """
    Handles command errors that occur during command invocation.

    Parameters:
        ctx (Context): The invocation context of the command.
        error (Exception): The error that occurred.

    Returns:
        None
    """
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Sorry, I don't recognize that command. Try using !chat, !message, !photo, or !clear.")
    else:
        logger.error(f"An error occurred: {error}")
        await ctx.send("An error occurred while processing your command.")

# Run the bot
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
