# AI-Powered Discord Bot Using Ollama & ComfyUI

This Discord bot integrates advanced AI capabilities, including text generation and image creation, to enhance user interactions on Discord servers. It leverages the Ollama API for text generation and ComfyUI for image generation, providing a versatile and engaging experience for users.

## Features

- **AI Chat**: Engage in conversations with the bot using natural language.
- **One-time Responses**: Get quick answers without affecting the conversation history.
- **Image Generation**: Create images based on text prompts.
- **Conversation Management**: Clear conversation history or continue ongoing dialogues.
- **Model Listing**: View available AI models for text generation.

## Prerequisites

- Python 3.7+
- Discord Developer Account and Bot Token
- Ollama API (local or remote)
- ComfyUI (for image generation)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/dustinwloring1988/ollama-discord-bot.git
   cd ollama-discord-bot
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   OLLAMA_API_URL=http://localhost:11434
   OLLAMA_MODEL=llama3.1
   COMFYUI_API_URL=http://127.0.0.1:8188
   ```

## Usage

Run the bot:
```
python bot.py
```

### Commands

- `!chat [message]`: Start or continue a conversation with the AI.
- `!message [text]`: Get a one-time response without affecting conversation history.
- `!photo [prompt]`: Generate an image based on the given prompt.
- `!clear`: Clear your conversation history with the bot.
- `!models`: List available AI models for text generation.

## Configuration

- Adjust the `OLLAMA_MODEL` in the `.env` file to use different text generation models.
- Modify the ComfyUI workflow in the `generate_image` function to customize image generation parameters.

## Logging

The bot uses Python's logging module. Logs are printed to the console with the following format:
```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This bot is for educational and personal use. Ensure compliance with Discord's Terms of Service and API usage guidelines.
