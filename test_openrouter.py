#!/usr/bin/env python3
"""
Test script for OpenRouter integration with Croq AI Assistant
This demonstrates how to use OpenRouter with multimodal content (text + images)
"""
import asyncio
import os
from models.base import OpenRouterClient, Message, MessageRole
from config import settings, ModelProvider

async def test_openrouter_text():
    """Test basic text generation with OpenRouter"""
    print("🔄 Testing OpenRouter text generation...")
    
    # Create OpenRouter client
    config = settings.get_model_config(ModelProvider.OPENROUTER)
    client = OpenRouterClient(config)
    
    # Create test messages
    messages = [
        Message(role=MessageRole.SYSTEM, content="You are a helpful AI assistant."),
        Message(role=MessageRole.USER, content="What is the capital of France? Please respond briefly.")
    ]
    
    try:
        async with client:
            response = await client.generate(messages)
            
        print(f"✅ Model: {response.model}")
        print(f"📝 Response: {response.content}")
        print(f"⚡ Latency: {response.latency:.2f}s")
        print(f"💰 Cost: ${response.cost:.4f}")
        print(f"📊 Tokens: {response.usage}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def test_openrouter_multimodal():
    """Test multimodal generation (text + image) with OpenRouter"""
    print("\n🔄 Testing OpenRouter multimodal generation...")
    
    config = settings.get_model_config(ModelProvider.OPENROUTER)
    client = OpenRouterClient(config)
    
    # Multimodal message with image URL
    messages = [
        Message(
            role=MessageRole.USER,
            content=[
                {
                    "type": "text",
                    "text": "What do you see in this image? Describe it briefly."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                }
            ]
        )
    ]
    
    try:
        async with client:
            response = await client.generate(messages)
            
        print(f"✅ Model: {response.model}")
        print(f"📝 Response: {response.content}")
        print(f"⚡ Latency: {response.latency:.2f}s")
        print(f"💰 Cost: ${response.cost:.4f}")
        print(f"📊 Tokens: {response.usage}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def test_openrouter_streaming():
    """Test streaming generation with OpenRouter"""
    print("\n🔄 Testing OpenRouter streaming...")
    
    config = settings.get_model_config(ModelProvider.OPENROUTER)
    client = OpenRouterClient(config)
    
    messages = [
        Message(role=MessageRole.USER, content="Write a short poem about AI and coding.")
    ]
    
    try:
        print("📝 Streaming response:")
        async with client:
            async for chunk in client.stream_generate(messages):
                print(chunk, end='', flush=True)
        print("\n✅ Streaming complete!")
        
    except Exception as e:
        print(f"❌ Streaming error: {e}")

def check_api_key():
    """Check if OpenRouter API key is configured"""
    if not settings.openrouter_api_key:
        print("⚠️  OpenRouter API key not found!")
        print("Please set OPENROUTER_API_KEY environment variable or add it to your .env file")
        print("Get your API key from: https://openrouter.ai/keys")
        return False
    print(f"✅ OpenRouter API key found: {settings.openrouter_api_key[:8]}...")
    return True

async def main():
    """Main test function"""
    print("🚀 OpenRouter Integration Test for Croq AI")
    print("=" * 50)
    
    if not check_api_key():
        return
    
    # Test basic text generation
    await test_openrouter_text()
    
    # Test multimodal (if supported by the model)
    await test_openrouter_multimodal()
    
    # Test streaming
    await test_openrouter_streaming()
    
    print("\n" + "=" * 50)
    print("🎉 OpenRouter integration tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
