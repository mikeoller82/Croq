# Croq 🚀

**Advanced AI Code Assistant**

Croq is a high-performance AI code assistant that combines multiple AI models with intelligent routing, advanced caching, comprehensive code analysis, and real-time streaming responses.

## 🎯 Key Features

### 🧠 Multi-Model Intelligence
- **Smart Router**: Automatically selects the best AI model for each task
- **Automatic Failover**: Seamless fallback between Claude, GPT-4, Groq, and Gemini
- **Concurrent Processing**: Race multiple models for fastest response
- **Cost Optimization**: Balance speed, quality, and cost automatically

### ⚡ Performance Optimizations
- **Multi-tier Caching**: Memory, disk, and Redis caching for instant responses
- **Async Architecture**: Non-blocking operations throughout
- **Request Deduplication**: Avoid redundant API calls
- **Smart Batching**: Group related requests for efficiency

### 🔍 Advanced Analysis
- **Code Quality Metrics**: Complexity, maintainability, documentation coverage
- **Security Scanning**: Built-in vulnerability detection with Bandit integration
- **Performance Analysis**: Identify bottlenecks and anti-patterns
- **AI-Powered Suggestions**: Get specific improvement recommendations

### 🎨 Superior UX
- **Real-time Streaming**: See code generation as it happens
- **Rich CLI Interface**: Beautiful terminal UI with syntax highlighting
- **Interactive Mode**: Conversational coding sessions
- **Multiple Output Formats**: Code, JSON, tables, and more

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
```bash
pip install -r requirements_optimized.txt
```

2. **Set up environment:**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Run Croq:**
```bash
python main.py --help
```

### Basic Usage

```bash
# Generate code
python main.py generate "Create a REST API with FastAPI"

# Analyze existing code
python main.py analyze mycode.py

# Interactive mode
python main.py interactive

# Stream code generation
python main.py generate "Build a web scraper" --stream

# Get performance stats
python main.py stats
```

## 📋 Commands

### Code Generation
```bash
# Basic generation
python main.py generate "Create a binary search function"

# Specify language
python main.py generate "Create a web server" --lang javascript

# With context
python main.py generate "Add error handling" --context "existing Flask app"

# Stream output
python main.py generate "Build a chatbot" --stream

# Save to file
python main.py generate "Database connection class" --output db.py
```

### Code Analysis
```bash
# Analyze file
python main.py analyze app.py

# Different language
python main.py analyze script.js --lang javascript

# JSON output
python main.py analyze code.py --format json
```

### Code Explanation
```bash
# General explanation
python main.py explain complex_algorithm.py

# Focus on performance
python main.py explain slow_function.py --focus performance

# Security analysis
python main.py explain auth.py --focus security
```

### System Management
```bash
# Health check
python main.py health

# Model status
python main.py models

# Performance statistics
python main.py stats

# Cache management
python main.py cache stats
python main.py cache clear
```

## ⚙️ Configuration

### API Keys
Get API keys from:
- **Claude**: [Anthropic Console](https://console.anthropic.com)
- **GPT-4**: [OpenAI Platform](https://platform.openai.com)  
- **Groq**: [Groq Console](https://console.groq.com) (Free tier available)
- **Gemini**: [Google AI Studio](https://aistudio.google.com)

### Environment Variables
```bash
# Required (at least one)
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here

# Optional optimizations
CACHE_ENABLED=true
MAX_CONCURRENT_REQUESTS=5
PRIMARY_MODEL=claude
```

## 🏗️ Architecture

```
Croq Optimized
├── 🧠 Multi-Model Router
│   ├── Claude (Advanced reasoning)
│   ├── GPT-4 (General purpose)
│   ├── Groq (Ultra-fast inference)
│   └── Gemini (Multimodal)
├── ⚡ Smart Cache
│   ├── Memory (L1 - fastest)
│   ├── Disk (L2 - persistent)  
│   └── Redis (L3 - distributed)
├── 🔍 Analysis Engine
│   ├── Code Quality Analyzer
│   ├── Security Scanner
│   └── Performance Profiler
└── 🎨 Rich Interface
    ├── CLI with syntax highlighting
    ├── Interactive mode
    └── Streaming responses
```

## 📊 Performance Comparison

| Feature | Croq Optimized | Claude Code | Gemini CLI |
|---------|----------------|-------------|------------|
| Multi-model routing | ✅ | ❌ | ❌ |
| Intelligent caching | ✅ | ❌ | ❌ |
| Concurrent requests | ✅ | ❌ | ❌ |
| Stream processing | ✅ | ✅ | ❌ |
| Security scanning | ✅ | ❌ | ❌ |
| Performance metrics | ✅ | ❌ | ❌ |
| Automatic failover | ✅ | ❌ | ❌ |
| Cost optimization | ✅ | ❌ | ❌ |

## 🎛️ Advanced Features

### Smart Model Selection
```python
# Automatically selects fastest model for simple tasks
croq generate "hello world function"  # → Groq (ultra-fast)

# Uses Claude for complex reasoning
croq generate "distributed system architecture"  # → Claude

# Concurrent racing for critical tasks
croq generate "production API" --concurrent  # → All models race
```

### Intelligent Caching
```python
# First request: hits API (2.5s)
croq generate "sorting algorithm"

# Second request: cache hit (0.05s) 
croq generate "sorting algorithm"  # 50x faster!

# Related requests benefit from semantic caching
croq generate "merge sort implementation"  # Partial cache hit
```

### Advanced Analysis
```python
# Comprehensive code analysis
croq analyze production_code.py
# Returns:
# - Quality score (0-10)
# - Security vulnerabilities
# - Performance bottlenecks  
# - Improvement suggestions
# - Complexity metrics
```

## 🔧 Development

### Project Structure
```
Croq/
├── config.py              # Configuration management
├── croq_optimized.py      # Main assistant class
├── cli.py                 # CLI interface
├── main.py                # Entry point
├── models/                # AI model integrations
│   ├── base.py           # Abstract model interface
│   └── router.py         # Intelligent routing
├── core/                 # Core systems
│   └── cache.py          # Multi-tier caching
├── utils/                # Utilities
│   ├── code_analysis.py  # Code quality analyzer
│   ├── security.py       # Security scanner
│   └── version_control.py # Git integration
└── requirements_optimized.txt
```

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=croq --cov-report=html
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Anthropic** for Claude API
- **OpenAI** for GPT-4 API  
- **Groq** for ultra-fast inference
- **Google** for Gemini API
- **Rich** for beautiful terminal UI

---

**Made with ❤️ for developers who demand excellence**

*Croq Optimized - Where AI meets performance*
