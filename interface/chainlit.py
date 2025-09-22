from dotenv import load_dotenv
load_dotenv()

import time
import chainlit as cl
import httpx
import asyncio
from typing import Optional
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/")

class ChatService:
    """Handles chat-related operations with async HTTP calls"""
    
    @staticmethod
    async def respond_to_chat(message: str) -> str:
        """Handle chat responses using async HTTP"""
        
        try:
            response = httpx.post(
                f"{API_URL}/chat",
                data={"message": message},
                timeout=30.0
            )

            if response.status_code == 200:
                print(response)
                raw_response = str(response.text).strip()
                
                # Unescape the response to convert \n to actual newlines
                if raw_response:
                    # Remove quotes if the response is wrapped in quotes
                    if raw_response.startswith('"') and raw_response.endswith('"'):
                        raw_response = raw_response[1:-1]
                    
                    # Convert escaped characters to actual characters
                    formatted_response = raw_response.replace('\\n', '\n').replace('\\"', '"').replace('\\*', '*')
                    
                    return formatted_response
                else:
                    return "**No response received from the API.**"
            else:
                return f"**API Error {response.status_code}:**\n\n```\n{response.text}\n```"
        
        except Exception as e:
            return f"**Error:** {str(e)}"

class DisplayService:
    """Handles display-related operations"""
    
    @staticmethod
    async def format_sentiment_response(sentiment_data: dict) -> str:
        """Format sentiment analysis response"""
        if not sentiment_data:
            return "Unable to analyze sentiment"
        
        # Format the sentiment analysis result
        sentiment = sentiment_data.get("sentiment", "Unknown")
        confidence = sentiment_data.get("confidence", 0.0)
        
        return f"**Sentiment**: {sentiment}\n**Confidence**: {confidence:.2%}"


class UIService:
    """Handles UI-related operations"""
    
    @staticmethod
    async def send_message(content: str, author="assistant"):
        """Send a simple message"""
        await cl.Message(
            content=content,
            author=author
        ).send()


async def run_typing_animation(msg: cl.Message):
    """Run typing animation until cancelled"""
    typing_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    frame_index = 0
    
    try:
        while True:  # Run indefinitely until cancelled
            frame = typing_frames[frame_index % len(typing_frames)]
            msg.content = f"{frame} Analyzing sentiment..."
            await msg.update()
            await asyncio.sleep(0.25)
            frame_index += 1
            
    except asyncio.CancelledError:
        # Animation was cancelled, this is expected
        print("🎬 Animation cancelled - API response received")
        raise

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    await cl.Message(
        content="**Sentiment Analysis**\n\n"
                "Hi there! 😊 I'm Sen, your friendly sentiment analysis assistant! I'm here to help you understand the emotions and feelings behind your words. Share anything with me and I'll cheerfully analyze the sentiment for you! 🌟\n\n"
                "Feel free to ask me about the sentiment of your text. Let's get started! 🚀",
        author="assistant"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """Main message handler with concurrent animation and API call"""
    # Create initial message for animation
    msg = cl.Message(content="", author="assistant")
    await msg.send()
    
    # Create concurrent tasks for animation and API call
    animation_task = asyncio.create_task(run_typing_animation(msg))
    api_task = asyncio.create_task(ChatService.respond_to_chat(message.content))
    
    try:
        # Wait for API response (this will complete first usually)
        response = await api_task
        
        # Cancel animation task since we have the response
        animation_task.cancel()
        
        # Wait a bit for graceful animation cancellation
        try:
            await asyncio.wait_for(animation_task, timeout=0.1)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        
    except Exception as e:
        # If API fails, cancel animation and show error
        animation_task.cancel()
        try:
            await asyncio.wait_for(animation_task, timeout=0.1)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        response = f"Error: {e}"
    
    # Update message with final response
    msg.content = response
    await msg.update()