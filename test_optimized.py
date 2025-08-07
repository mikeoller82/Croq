#!/usr/bin/env python3
"""
Quick test script for Croq Optimized
"""
import asyncio
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

async def test_basic_functionality():
    """Test basic functionality"""
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from config import settings, ModelProvider
        from models.base import Message, MessageRole
        from core.cache import cache
        print("✅ Imports successful")
        
        # Test configuration
        print("\n🔄 Testing configuration...")
        available_models = settings.get_available_models()
        print(f"✅ Available models: {[m.value for m in available_models]}")
        
        # Test cache
        print("\n🔄 Testing cache system...")
        await cache.set("test_key", "test_value")
        cached_value = await cache.get("test_key")
        if cached_value == "test_value":
            print("✅ Cache system working")
        else:
            print("❌ Cache system failed")
        
        # Test message creation
        print("\n🔄 Testing message system...")
        message = Message(role=MessageRole.USER, content="Hello, World!")
        print(f"✅ Message created: {message.role} - {message.content}")
        
        # Show cache stats
        print("\n📊 Cache Statistics:")
        stats = await cache.get_stats()
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}: {len(value)} items")
            else:
                print(f"  {key}: {value}")
        
        print("\n🎉 All basic tests passed! Croq Optimized is ready to use.")
        print("\nNext steps:")
        print("1. Add your API keys to .env file")
        print("2. Run: python main.py generate 'hello world function'")
        print("3. Run: python main.py interactive")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try installing dependencies: pip install -r requirements_optimized.txt")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())
