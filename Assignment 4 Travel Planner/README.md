# 🌏 Enhanced AI Travel Agent & Real-Time Hotel Planner

A comprehensive AI-powered travel planning application with real-time data integration, weather forecasts, budget analysis, and personalized itineraries.

## 🚀 Features

### Core Features
- **AI-Powered Itinerary Generation**: Detailed day-by-day travel plans
- **Real-Time Weather Data**: Current weather and 5-day forecasts
- **Budget Planning**: Comprehensive budget breakdown with currency conversion
- **Hotel Recommendations**: Real-time options from TripAdvisor and Google Places
- **Restaurant Suggestions**: Popular dining options with ratings

### Advanced Features
- **🔍 Real-Time Data Sources**: DuckDuckGo search for current prices and insights
- **🚀 Enhanced LangGraph Workflow**: Advanced AI agent with multiple search APIs
- **📊 Multi-Source Cost Analysis**: Real-time cost validation
- **📄 Export Options**: CSV and Markdown export
- **🎯 Planning Styles**: Balanced, Budget-Friendly, Luxury, Adventure, Cultural, Relaxation
- **⏱️ Trip Pace**: Relaxed, Moderate, Fast-paced options

## 🛠️ Installation

1. **Clone and navigate to project**
   ```bash
   cd Agenticbatch2/Assignment
   ```

2. **Create virtual environment**
   ```bash
   python -m venv TPA
   TPA\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install gradio openai googlemaps requests duckduckgo-search langchain langgraph
   ```

4. **Set API keys** (in the code or environment variables)
   - OpenAI API Key
   - Google Places API Key
   - RapidAPI Key
   - Weather API Key

5. **Run the application**
   ```bash
   python travel_planner_gradio.py
   ```

6. **Access web interface**: `http://127.0.0.1:7860`

## 📋 Usage

### Basic Planning
1. Enter destination city
2. Set start and end dates
3. Specify number of travelers and interests
4. Set budget and currency
5. Choose planning style and pace
6. Enable desired features
7. Click "Plan My Trip"

### Planning Options
- **Planning Styles**: Balanced, Budget-Friendly, Luxury, Adventure, Cultural, Relaxation
- **Trip Pace**: Relaxed, Moderate, Fast-paced
- **Enhanced Features**: LangGraph workflow, real-time data, markdown export

## 📊 Output Sections

1. **Weather Forecast**: Current conditions and trip forecast
2. **Budget Summary**: Detailed cost breakdown and analysis
3. **Hotel Options**: TripAdvisor and Google Places recommendations
4. **Restaurant Suggestions**: Popular dining options
5. **AI Itinerary**: Detailed day-by-day schedule
6. **Real-Time Insights**: DuckDuckGo search results
7. **Enhanced Analysis**: Advanced AI recommendations (optional)

## 🔧 Configuration

### Required API Keys
```python
OPENAI_API_KEY = "your_openai_key"
WEATHER_API_KEY = "your_openweathermap_key"
GOOGLE_API_KEY = "your_google_places_key"
RAPIDAPI_KEY = "your_rapidapi_key"
```

## 🐛 Troubleshooting

### Common Issues
- **API Key Errors**: Verify all keys are correctly set
- **Network Timeouts**: App includes retry logic and fallbacks
- **Weather Data Missing**: Fallback information provided
- **Search Failures**: Network issues or rate limiting

### Error Handling
- Graceful fallbacks for API failures
- Timeout handling for external requests
- Comprehensive error messages
- Fallback data for popular cities

## 📁 Project Structure

```
Agenticbatch2/Assignment/
├── travel_planner_gradio.py    # Main application
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── TPA/                        # Virtual environment
```

## 🚀 Technologies

- **Frontend**: Gradio
- **AI**: OpenAI GPT-4o, LangGraph, LangChain
- **APIs**: OpenWeatherMap, Google Places, TripAdvisor, DuckDuckGo
- **Data**: JSON, CSV, Markdown export

## 🎯 Use Cases

- **Travelers**: Complete trip planning with real-time data
- **Travel Agents**: Professional travel plan generation
- **Developers**: Extensible architecture for customization

## 📝 License

For educational and personal use. Respect API terms of service.

---

**🌏 Enhanced AI Travel Agent** - Smart, comprehensive travel planning powered by AI! 