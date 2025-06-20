# ===================== IMPORTS =====================
# Standard and third-party library imports
import gradio as gr
import requests
from openai import OpenAI
import os
import googlemaps
import json
import csv
from datetime import datetime, timedelta
from dateutil import parser as date_parser
from duckduckgo_search import DDGS
import threading
import time

# LangGraph and LangChain imports for enhanced agent workflow
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Dict, List, Any, Literal

# Additional search APIs
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import SerpAPIWrapper, GoogleSerperAPIWrapper
from langchain_google_community import GooglePlacesAPIWrapper, GooglePlacesTool

# ============== CONFIGURATION ==============
# API keys and configuration class
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
CURRENCY_API_KEY = os.getenv("CURRENCY_API_KEY", "")
OPENAI_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

class Config:
    """Configuration class for all API keys and settings"""
    def __init__(self):
        self.weather_api_key = WEATHER_API_KEY
        self.currency_api_key = CURRENCY_API_KEY
        self.openai_api_key = OPENAI_API_KEY
        self.google_api_key = GOOGLE_API_KEY
        self.rapidapi_key = RAPIDAPI_KEY
        self.serpapi_key = SERPAPI_KEY
        self.serper_api_key = SERPER_API_KEY

# ============== CLIENT INITIALIZATION ==============
# OpenAI and Google Maps clients
client = OpenAI(api_key=OPENAI_API_KEY)
gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# ============== CORE FUNCTIONS ==============
# Core business logic for fetching data, processing information, and generating travel plans.

def get_current_weather(city: str) -> str | None:
    """
    Fetches the current weather for a given city from OpenWeatherMap API.

    Args:
        city: The name of the city.

    Returns:
        A formatted string with weather description and temperature, or None if an error occurs.
    """
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10).json()
        if response.get("cod") != 200:
            return None
        return f"{response['weather'][0]['description'].title()}, {response['main']['temp']}°C"
    except requests.exceptions.RequestException as e:
        print(f"Weather API network error: {e}")
        return None
    except Exception as e:
        print(f"Weather API error: {e}")
        return None

def get_weather_forecast(city: str) -> dict | None:
    """
    Fetches a 5-day weather forecast for a given city from OpenWeatherMap API.

    Args:
        city: The name of the city.

    Returns:
        A dictionary with dates as keys and formatted weather strings as values, or None on error.
    """
    try:
        url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url, timeout=10).json()
        if response.get("cod") != "200":
            return None
        forecast_list = response["list"]
        daily_forecast = {}
        for item in forecast_list:
            date = item["dt_txt"].split(" ")[0]
            if date not in daily_forecast:
                daily_forecast[date] = f"{item['weather'][0]['description'].title()}, {item['main']['temp']}°C"
        return daily_forecast
    except requests.exceptions.RequestException as e:
        print(f"Weather forecast API network error: {e}")
        return None
    except Exception as e:
        print(f"Weather forecast API error: {e}")
        return None

def get_exchange_rate(from_currency: str, to_currency: str) -> float | None:
    """
    Fetches the exchange rate between two currencies from ExchangeRate-API.

    Args:
        from_currency: The base currency code (e.g., "USD").
        to_currency: The target currency code (e.g., "INR").

    Returns:
        The exchange rate as a float, or None if an error occurs.
    """
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        response = requests.get(url, timeout=10).json()
        return response["rates"].get(to_currency, None)
    except requests.exceptions.RequestException as e:
        print(f"Exchange rate API network error: {e}")
        return None
    except Exception as e:
        print(f"Exchange rate API error: {e}")
        return None

def convert_currency(amount: float, rate: float) -> float:
    """
    Converts an amount from one currency to another using a given rate.

    Args:
        amount: The amount to convert.
        rate: The exchange rate.

    Returns:
        The converted amount, rounded to 2 decimal places.
    """
    return round(amount * rate, 2)

def orchestrate_itinerary(city: str, start_date: str, end_date: str, interests: str) -> str:
    """
    Generates a detailed daily itinerary using an LLM (OpenAI's GPT-4o).

    Args:
        city: The destination city.
        start_date: The start date of the trip.
        end_date: The end date of the trip.
        interests: A string of user interests to tailor the itinerary.

    Returns:
        A JSON string representing the daily itinerary.
    """
    prompt = f"""
    You are a travel assistant. Generate a comprehensive daily itinerary for a trip to {city}
    starting on {start_date} and ending on {end_date}.
    Interests: {interests}.
    
    For each day, include ALL of the following activities:
    - Morning activity (site, attraction, or experience)
    - Lunch suggestion (restaurant or local cuisine)
    - Afternoon activity (site, attraction, or experience)
    - Evening activity (site, attraction, or experience)
    - Dinner suggestion (restaurant or local cuisine)
    - Night activity (entertainment, nightlife, or relaxation)

    Guidelines:
    - Do not repeat the same places across different days
    - Vary the types of activities (cultural, outdoor, food, entertainment)
    - Consider the interests provided: {interests}
    - Make activities suitable for the time of day
    - Include both tourist attractions and local experiences
    - Provide specific restaurant names when possible

    IMPORTANT: You must respond with ONLY valid JSON. Do not include any text before or after the JSON.
    The JSON must follow this exact structure:

    {{
        "days": [
            {{
                "day": 1,
                "morning": "activity description",
                "lunch": "restaurant or cuisine suggestion",
                "afternoon": "activity description",
                "evening": "activity description",
                "dinner": "restaurant or cuisine suggestion",
                "night": "activity description"
            }}
        ]
    }}

    Return ONLY the JSON object, nothing else.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

def get_tripadvisor_hotels(city: str, check_in: str, check_out: str, adults: int, min_price: float, max_price: float) -> list:
    """
    Searches for hotels on TripAdvisor via RapidAPI within a specified price range.

    Args:
        city: The destination city.
        check_in: The check-in date.
        check_out: The check-out date.
        adults: The number of adults.
        min_price: The minimum price per night.
        max_price: The maximum price per night.

    Returns:
        A list of formatted strings with hotel names and prices, or an error message.
    """
    try:
        # Step 1: Get location_id for the city
        url = "https://travel-advisor.p.rapidapi.com/locations/search"
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }
        params = {
            "query": city,
            "limit": "5",
            "offset": "0",
            "units": "km",
            "currency": "USD",
            "sort": "relevance",
            "lang": "en_US"
        }
        
        # Step 1: Get location_id for the city
        loc_resp = requests.get(url, headers=headers, params=params)
        loc_data = loc_resp.json()
        location_id = None
        for d in loc_data.get("data", []):
            if d.get("result_type") == "geos":
                location_id = d.get("result_object", {}).get("location_id")
                break
        if not location_id:
            return []
        
        # Step 2: Search for hotels in that location
        hotels_url = "https://travel-advisor.p.rapidapi.com/hotels/list"
        hotels_params = {
            "location_id": location_id,
            "adults": str(adults),
            "checkin": check_in,
            "checkout": check_out,
            "currency": "USD",
            "order": "PRICE",
            "limit": "10",
            "lang": "en_US"
        }
        
        # Only add price filters if they're reasonable
        if min_price > 0 and max_price > min_price:
            hotels_params["min_price"] = str(int(min_price))
            hotels_params["max_price"] = str(int(max_price))
        
        hotels_resp = requests.get(hotels_url, headers=headers, params=hotels_params)
        hotels_data = hotels_resp.json()
        hotels = hotels_data.get("data", [])
        return [
            f"{h.get('name', 'N/A')} — {h.get('price', 'N/A')}"
            for h in hotels if h.get("name") and h.get("price")
        ] if hotels else []
        
    except Exception as e:
        return [f"Error fetching hotel data: {str(e)}"]

def get_google_places(city: str, place_type: str, max_results: int = 5) -> list:
    """
    Searches for places (hotels or restaurants) using the Google Places API.

    Args:
        city: The destination city.
        place_type: The type of place to search for ('lodging' or 'restaurant').
        max_results: The maximum number of results to return.

    Returns:
        A list of dictionaries, each containing details about a place.
    """
    try:
        # First, geocode the city to get coordinates
        geocode_result = gmaps.geocode(city)
        if not geocode_result:
            return []
        
        lat = geocode_result[0]['geometry']['location']['lat']
        lng = geocode_result[0]['geometry']['location']['lng']
        
        # Search for places
        places_result = gmaps.places_nearby(
            location=(lat, lng),
            radius=5000,  # 5km radius
            type=place_type
        )
        
        places = []
        for place in places_result.get('results', [])[:max_results]:
            name = place.get('name', 'N/A')
            rating = place.get('rating', 'N/A')
            price_level = place.get('price_level', 'N/A')
            address = place.get('vicinity', 'N/A')
            
            # Convert price level to readable format and estimated cost
            price_text = 'N/A'
            estimated_cost = 'N/A'
            
            if price_level == 0:
                price_text = 'Free'
                estimated_cost = 0
            elif price_level == 1:
                price_text = '$'
                estimated_cost = 15 if place_type == 'restaurant' else 50
            elif price_level == 2:
                price_text = '$$'
                estimated_cost = 30 if place_type == 'restaurant' else 100
            elif price_level == 3:
                price_text = '$$$'
                estimated_cost = 60 if place_type == 'restaurant' else 200
            elif price_level == 4:
                price_text = '$$$$'
                estimated_cost = 100 if place_type == 'restaurant' else 400
            
            places.append({
                'name': name,
                'rating': rating,
                'price_level': price_text,
                'address': address,
                'estimated_cost': estimated_cost
            })
        
        return places
        
    except Exception as e:
        return []

def plan_trip(city, start_date, end_date, num_people, interests, trip_budget, currency_code):
    """
    Main function to orchestrate the entire trip planning process.
    It fetches data from various sources, generates an itinerary, calculates the budget,
    and formats all information for display in the UI.

    Args:
        city: Destination city.
        start_date: Trip start date.
        end_date: Trip end date.
        num_people: Number of travelers.
        interests: User's interests.
        trip_budget: Total budget in USD.
        currency_code: User's local currency.

    Returns:
        A tuple containing formatted strings for weather, budget, hotels, Google Places,
        the itinerary (JSON), and DuckDuckGo insights.
    """
    
    # --- 1. Input Validation ---
    if not all([city, start_date, end_date, num_people, trip_budget, currency_code]):
        return "Please fill out all required inputs.", "", "", "", "", ""
    
    # Parse dates from strings to ensure they are valid
    try:
        start_date_obj = date_parser.parse(start_date).date()
        end_date_obj = date_parser.parse(end_date).date()
    except:
        return "Please enter valid dates in YYYY-MM-DD format (e.g., 2024-06-15).", "", "", "", "", ""
    
    if end_date_obj < start_date_obj:
        return "End date must be after start date.", "", "", "", "", ""
    
    try:
        # --- 2. Initial Calculations ---
        num_days = (end_date_obj - start_date_obj).days + 1
        
        # --- 3. Data Fetching and Fallbacks ---
        
        # Get weather data with fallback for network errors
        current_weather = get_current_weather(city)
        if not current_weather:
            # Provide fallback weather info instead of failing completely
            current_weather = f"Weather data temporarily unavailable for {city}. Please check weather forecasts before your trip."
        
        forecast = get_weather_forecast(city)
        if not forecast:
            # Provide fallback forecast info for all days if API fails
            from datetime import timedelta
            fallback_forecast = {}
            current_date = start_date_obj
            while current_date <= end_date_obj:
                date_str = current_date.strftime("%Y-%m-%d")
                fallback_forecast[date_str] = "Weather forecast temporarily unavailable. Please check weather services before your trip."
                current_date += timedelta(days=1)
            forecast = fallback_forecast
        
        # Get exchange rate, trying DuckDuckGo first, then a reliable API, then a static fallback
        duckduckgo_rate = get_currency_exchange_duckduckgo("USD", currency_code)
        if duckduckgo_rate:
            exchange_rate = duckduckgo_rate
            exchange_source = "DuckDuckGo real-time search"
        else:
            exchange_rate = get_exchange_rate("USD", currency_code)
            exchange_source = "Exchange Rate API"
        
        # If both online sources fail, use a hardcoded fallback rate for major currencies
        if exchange_rate is None:
            # Common fallback rates for major currencies
            fallback_rates = {
                'INR': 83.0, 'EUR': 0.92, 'GBP': 0.79, 'JPY': 150.0, 
                'CAD': 1.35, 'AUD': 1.52, 'CHF': 0.88, 'CNY': 7.2
            }
            exchange_rate = fallback_rates.get(currency_code.upper(), 1.0)
            exchange_source = "Fallback estimate"
        
        # Final check for exchange rate to prevent crashes
        if exchange_rate is None:
            return f"Could not fetch exchange rate for {currency_code}. Please check the currency code.", "", "", "", "", ""
        
        # --- 4. Budget Calculation ---
        accommodation_budget = trip_budget * 0.5
        per_night_budget = accommodation_budget / num_days
        lower_bound = per_night_budget * 0.9
        upper_bound = per_night_budget * 1.1
        
        food_budget = trip_budget * 0.2
        travel_budget = trip_budget * 0.1
        leisure_budget = trip_budget * 0.2
        
        local_total_budget = convert_currency(trip_budget, exchange_rate)
        
        # --- 5. AI Itinerary Generation ---
        itinerary_text = orchestrate_itinerary(city, start_date_obj, end_date_obj, interests)
        
        # Itinerary is kept as raw JSON for flexibility in the UI (e.g., CSV export)
        formatted_itinerary = itinerary_text
        
        # --- 6. Real-Time Cost & Hotel Analysis ---
        
        # Get comprehensive cost analysis using multiple online sources
        cost_analysis = get_comprehensive_cost_analysis(city, trip_budget, num_days, num_people, start_date_obj, end_date_obj)
        
        # Use comprehensive estimates if available, otherwise fall back to basic estimates
        if cost_analysis and cost_analysis["final_estimates"]:
            daily_costs = cost_analysis["final_estimates"]
            data_sources = daily_costs.get("data_sources", [])
        else:
            # Fallback to basic, static estimates if real-time analysis fails
            daily_costs = estimate_daily_costs(city, trip_budget, num_days, num_people)
            data_sources = ["Basic estimates"]
        
        # Get hotel options from TripAdvisor based on the calculated budget
        hotel_options = get_tripadvisor_hotels(
            city=city,
            check_in=start_date,
            check_out=end_date,
            adults=num_people,
            min_price=lower_bound,
            max_price=upper_bound
        )
        
        # Get hotel and restaurant data from Google Places
        google_hotels = get_google_places(city, "lodging", max_results=5)
        google_restaurants = get_google_places(city, "restaurant", max_results=5)
        
        # --- 7. Formatting Outputs for UI Display ---
        
        # Format weather forecast to be user-friendly, showing all trip dates
        weather_info = f"Current: {current_weather}\n\nForecast during trip:\n"
        
        # Generate all dates for the trip to ensure the forecast covers the entire duration
        from datetime import timedelta
        current_date = start_date_obj
        forecast_days = 0
        
        while current_date <= end_date_obj:
            date_str = current_date.strftime("%Y-%m-%d")
            
            # Check if we have forecast data for this date and display it, or show a fallback
            if date_str in forecast:
                weather_info += f"{date_str}: {forecast[date_str]}\n"
                forecast_days += 1
            else:
                # If no forecast data available, show a fallback message
                weather_info += f"{date_str}: Forecast data not available\n"
            
            current_date += timedelta(days=1)
        
        # If no forecast data was found at all, show a general informational message
        if forecast_days == 0:
            weather_info += f"⚠️ Detailed forecast not available for all dates.\n"
            weather_info += f"💡 Check weather services closer to your travel dates for accurate forecasts.\n"
        
        # Format insights gathered from DuckDuckGo search
        duckduckgo_info = ""
        if cost_analysis and cost_analysis.get('duckduckgo_insights'):
            insights = cost_analysis['duckduckgo_insights']
            duckduckgo_info = "🔍 Real-Time Insights (DuckDuckGo Search):\n"
            duckduckgo_info += "=" * 50 + "\n\n"
            
            has_content = False
            
            # Local insights
            if insights.get('local_insights') and insights['local_insights']:
                duckduckgo_info += "📍 Local Insights:\n"
                for query, results in insights['local_insights'].items():
                    if results and len(results) > 0:
                        # Show complete sentences, not truncated
                        insight_text = results[0]
                        # Clean up the query name for display
                        query_display = query.replace(city, '').strip()
                        if query_display:
                            duckduckgo_info += f"• **{query_display}**: {insight_text}\n"
                        else:
                            duckduckgo_info += f"• {insight_text}\n"
                        has_content = True
                if has_content:
                    duckduckgo_info += "\n"
            
            # Weather insights
            if insights.get('weather_insights') and insights['weather_insights']:
                duckduckgo_info += "🌤️ Weather Insights:\n"
                for query, results in insights['weather_insights'].items():
                    if results and len(results) > 0:
                        # Show complete sentences, not truncated
                        insight_text = results[0]
                        # Clean up the query name for display
                        query_display = query.replace(city, '').strip()
                        if query_display:
                            duckduckgo_info += f"• **{query_display}**: {insight_text}\n"
                        else:
                            duckduckgo_info += f"• {insight_text}\n"
                        has_content = True
                if has_content:
                    duckduckgo_info += "\n"
            
            # Travel alerts
            if insights.get('travel_alerts') and insights['travel_alerts']:
                duckduckgo_info += "⚠️ Travel Alerts:\n"
                for query, results in insights['travel_alerts'].items():
                    if results and len(results) > 0:
                        # Show complete sentences, not truncated
                        insight_text = results[0]
                        # Clean up the query name for display
                        query_display = query.replace(city, '').strip()
                        if query_display:
                            duckduckgo_info += f"• **{query_display}**: {insight_text}\n"
                        else:
                            duckduckgo_info += f"• {insight_text}\n"
                        has_content = True
                if has_content:
                    duckduckgo_info += "\n"
            
            # Price insights
            if insights.get('hotel_prices') and insights['hotel_prices']:
                duckduckgo_info += "🏨 Hotel Price Insights:\n"
                for price_info in insights['hotel_prices'][:2]:
                    if price_info:
                        # Show complete sentences, not truncated
                        duckduckgo_info += f"• {price_info}\n"
                        has_content = True
                if has_content:
                    duckduckgo_info += "\n"
            
            if insights.get('food_prices') and insights['food_prices']:
                duckduckgo_info += "🍽️ Food Price Insights:\n"
                for price_info in insights['food_prices'][:2]:
                    if price_info:
                        # Show complete sentences, not truncated
                        duckduckgo_info += f"• {price_info}\n"
                        has_content = True
                if has_content:
                    duckduckgo_info += "\n"
            
            # If no content was found, show a fallback message
            if not has_content:
                duckduckgo_info = "🔍 Real-Time Insights (DuckDuckGo Search):\n"
                duckduckgo_info += "=" * 50 + "\n\n"
                duckduckgo_info += "ℹ️ No real-time insights available at the moment.\n"
                duckduckgo_info += "This could be due to:\n"
                duckduckgo_info += "• Network connectivity issues\n"
                duckduckgo_info += "• Search rate limiting\n"
                duckduckgo_info += "• No recent data for this destination\n\n"
                duckduckgo_info += "💡 Try refreshing or check back later for updated insights.\n"
        else:
            # Fallback when no DuckDuckGo insights are available
            duckduckgo_info = "🔍 Real-Time Insights (DuckDuckGo Search):\n"
            duckduckgo_info += "=" * 50 + "\n\n"
            duckduckgo_info += "ℹ️ DuckDuckGo insights are being loaded...\n"
            duckduckgo_info += "If this message persists, it may indicate:\n"
            duckduckgo_info += "• Network connectivity issues\n"
            duckduckgo_info += "• Search service temporarily unavailable\n"
            duckduckgo_info += "• Rate limiting from search provider\n\n"
            duckduckgo_info += "💡 The rest of your travel plan is still fully functional!\n"
        
        # Format the detailed budget summary, handling different data source keys for robustness
        budget_summary = f"""💰 Budget Summary
Total Trip Budget: ${trip_budget:,.2f} USD
Exchange Rate: 1 USD ≈ {exchange_rate:.2f} {currency_code} (Source: {exchange_source})
Total Budget in {currency_code}: {currency_code} {local_total_budget:,.2f}

Budget Breakdown:
• Accommodation: ${accommodation_budget:,.2f} USD
• Food: ${food_budget:,.2f} USD  
• Travel: ${travel_budget:,.2f} USD
• Leisure: ${leisure_budget:,.2f} USD

Per Night Budget: ${per_night_budget:.2f} USD (Range: ${lower_bound:.2f} - ${upper_bound:.2f})

📊 Real-Time Cost Analysis for {city}:
• Daily Accommodation: ${daily_costs.get('daily_accommodation', daily_costs.get('accommodation_per_night', 0)):.2f} USD
• Daily Food: ${daily_costs.get('daily_food', daily_costs.get('food_per_day', 0)):.2f} USD
• Daily Transport: ${daily_costs.get('transport_per_day', 8 * num_people):.2f} USD
• Total Daily Cost: ${daily_costs.get('total_daily', daily_costs.get('total_per_day', 0)):.2f} USD
• Total Trip Cost: ${daily_costs.get('total_daily', daily_costs.get('total_per_day', 0)) * num_days:.2f} USD

💡 Budget Analysis:
• Your daily budget: ${trip_budget / num_days:.2f} USD
• Estimated daily cost: ${daily_costs.get('total_daily', daily_costs.get('total_per_day', 0)):.2f} USD
• Budget status: {'✅ Within budget' if (trip_budget / num_days) >= daily_costs.get('total_daily', daily_costs.get('total_per_day', 0)) else '⚠️ May exceed budget'}

🔍 Data Sources: {', '.join(data_sources)}"""
        
        # Format hotel options from TripAdvisor
        hotel_info = "🏨 Real-Time Hotel Options (TripAdvisor):\n"
        if hotel_options:
            for hotel in hotel_options:
                hotel_info += f"• {hotel}\n"
        else:
            hotel_info += "No hotels found within your budget range.\n"
        
        # Format hotel and restaurant data from Google Places
        google_info = "🏨 Google Places - Popular Hotels:\n"
        if google_hotels:
            for hotel in google_hotels:
                cost_info = f"~${hotel['estimated_cost']}/night" if hotel['estimated_cost'] != 'N/A' else "Price N/A"
                google_info += f"• {hotel['name']} - {hotel['address']} | ⭐ {hotel['rating']} | 💰 {hotel['price_level']} | {cost_info}\n"
        else:
            google_info += "No hotels found via Google Places API.\n"
        
        google_info += "\n🍽️ Google Places - Popular Restaurants:\n"
        if google_restaurants:
            for restaurant in google_restaurants:
                cost_info = f"~${restaurant['estimated_cost']}/meal" if restaurant['estimated_cost'] != 'N/A' else "Price N/A"
                google_info += f"• {restaurant['name']} - {restaurant['address']} | ⭐ {restaurant['rating']} | 💰 {restaurant['price_level']} | {cost_info}\n"
        else:
            google_info += "No restaurants found via Google Places API.\n"
        
        # --- 8. Return All Results ---
        return weather_info, budget_summary, hotel_info, google_info, formatted_itinerary, duckduckgo_info
        
    except Exception as e:
        # Global error handler for the planning function to prevent crashes
        return f"Error planning trip: {str(e)}", "", "", "", "", ""

def format_itinerary(itinerary_text):
    """
    Formats a JSON itinerary into a human-readable, fixed-width table format.
    This provides a clean, professional layout in the UI.

    Args:
        itinerary_text: The raw JSON string of the itinerary.

    Returns:
        A formatted string, or an error message if parsing fails.
    """
    try:
        # Parse the JSON
        itinerary_data = json.loads(itinerary_text)
        
        formatted_output = "🗓️ YOUR PERSONALIZED ITINERARY\n"
        formatted_output += "=" * 80 + "\n\n"
        
        # Create table header
        formatted_output += f"{'Day':<4} {'Time':<12} {'Activity':<60}\n"
        formatted_output += "-" * 80 + "\n"
        
        for day_data in itinerary_data.get("days", []):
            day_num = day_data.get("day", "N/A")
            
            # Morning activity
            morning = day_data.get("morning", "No activity planned")
            formatted_output += f"{day_num:<4} {'🌅 Morning':<12} {morning:<60}\n"
            
            # Lunch
            lunch = day_data.get("lunch", "No lunch planned")
            formatted_output += f"{'':<4} {'🍽️ Lunch':<12} {lunch:<60}\n"
            
            # Afternoon activity
            afternoon = day_data.get("afternoon", "No activity planned")
            formatted_output += f"{'':<4} {'☀️ Afternoon':<12} {afternoon:<60}\n"
            
            # Evening activity
            evening = day_data.get("evening", "No activity planned")
            formatted_output += f"{'':<4} {'🌆 Evening':<12} {evening:<60}\n"
            
            # Dinner
            dinner = day_data.get("dinner", "No dinner planned")
            formatted_output += f"{'':<4} {'🍴 Dinner':<12} {dinner:<60}\n"
            
            # Night activity
            night = day_data.get("night", "No activity planned")
            formatted_output += f"{'':<4} {'🌙 Night':<12} {night:<60}\n"
            
            # Add separator between days
            formatted_output += "-" * 80 + "\n"
        
        return formatted_output
        
    except json.JSONDecodeError:
        # If JSON parsing fails, return the original text with a note
        return f"⚠️ Could not parse itinerary format. Here's the raw output:\n\n{itinerary_text}"
    except Exception as e:
        return f"⚠️ Error formatting itinerary: {str(e)}\n\nRaw output:\n{itinerary_text}"

def estimate_daily_costs(city, trip_budget, num_days, num_people):
    """
    Provides basic, static daily cost estimates as a fallback.
    These estimates are used when real-time API calls fail.

    Args:
        city: The destination city.
        trip_budget: The total budget for the trip.
        num_days: The number of days for the trip.
        num_people: The number of travelers.

    Returns:
        A dictionary with estimated daily costs for accommodation and food.
    """
    try:
        # Get cost of living data for the city using a simple estimation
        # This is a basic estimation - in a real app you'd use a cost of living API
        
        # Base estimates per person per day (in USD)
        base_estimates = {
            'dubai': {'accommodation': 120, 'food': 40},
            'london': {'accommodation': 150, 'food': 50},
            'new york': {'accommodation': 180, 'food': 60},
            'paris': {'accommodation': 140, 'food': 45},
            'tokyo': {'accommodation': 130, 'food': 35},
            'singapore': {'accommodation': 140, 'food': 40},
            'bangkok': {'accommodation': 60, 'food': 20},
            'bali': {'accommodation': 80, 'food': 25},
            'mumbai': {'accommodation': 70, 'food': 15},
            'delhi': {'accommodation': 65, 'food': 15},
            'bangalore': {'accommodation': 75, 'food': 18},
            'chennai': {'accommodation': 60, 'food': 15},
            'kolkata': {'accommodation': 55, 'food': 12},
            'hyderabad': {'accommodation': 65, 'food': 16},
            'pune': {'accommodation': 60, 'food': 15}
        }
        
        city_lower = city.lower().strip()
        
        # Get estimates for the city, or use default if not found
        if city_lower in base_estimates:
            estimates = base_estimates[city_lower]
        else:
            # Default estimates for unknown cities
            estimates = {'accommodation': 100, 'food': 30}
        
        # Calculate daily costs
        daily_accommodation = estimates['accommodation'] * num_people
        daily_food = estimates['food'] * num_people
        
        # Adjust based on budget level
        total_daily_cost = daily_accommodation + daily_food
        budget_per_day = trip_budget / num_days
        
        # If budget is higher than estimated, adjust estimates upward
        if budget_per_day > total_daily_cost * 1.5:
            adjustment_factor = min(budget_per_day / total_daily_cost, 2.0)
            daily_accommodation *= adjustment_factor
            daily_food *= adjustment_factor
        # If budget is lower than estimated, adjust estimates downward
        elif budget_per_day < total_daily_cost * 0.7:
            adjustment_factor = max(budget_per_day / total_daily_cost, 0.5)
            daily_accommodation *= adjustment_factor
            daily_food *= adjustment_factor
        
        return {
            'daily_accommodation': round(daily_accommodation, 2),
            'daily_food': round(daily_food, 2),
            'total_daily': round(daily_accommodation + daily_food, 2),
            'city_known': city_lower in base_estimates
        }
        
    except Exception as e:
        # Fallback estimates
        return {
            'daily_accommodation': round(100 * num_people, 2),
            'daily_food': round(30 * num_people, 2),
            'total_daily': round(130 * num_people, 2),
            'city_known': False
        }

def get_cost_of_living_data(city):
    """
    Placeholder function to get cost of living data.
    In a real application, this would call a dedicated API like Numbeo.

    Args:
        city: The destination city.

    Returns:
        A dictionary with cost of living indices.
    """
    try:
        # This would ideally use a cost of living API
        # For now, return basic structure
        return {
            'city': city,
            'cost_index': 'N/A',
            'rent_index': 'N/A',
            'restaurant_index': 'N/A',
            'groceries_index': 'N/A'
        }
    except:
        return None

def get_real_time_costs_llm(city, num_people, num_days):
    """
    Uses an LLM to research and estimate real-time travel costs.
    This acts as a smart research agent to find up-to-date pricing.

    Args:
        city: The destination city.
        num_people: The number of travelers.
        num_days: The number of days for the trip.

    Returns:
        A dictionary with LLM-estimated costs, or None on error.
    """
    try:
        prompt = f"""
        Research and provide current average daily costs for {city} in 2024. 
        Focus on mid-range options for {num_people} person(s) for {num_days} days.
        
        Please provide estimates for:
        1. Accommodation (mid-range hotel per night)
        2. Food (3 meals per day - breakfast, lunch, dinner)
        3. Local transportation (public transport, taxis, rideshares)
        4. Activities/Entertainment (attractions, tours, experiences)
        
        Consider current inflation and 2024 prices. Provide realistic, up-to-date estimates.
        
        Respond in this exact JSON format:
        {{
            "accommodation_per_night": number,
            "food_per_day": number,
            "transport_per_day": number,
            "activities_per_day": number,
            "total_per_day": number,
            "source_notes": "brief description of sources considered",
            "currency": "USD"
        }}
        
        Only return valid JSON, no additional text.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        return None

def get_cost_of_living_api(city):
    """
    Fetches cost of living data from an online API (Numbeo demo).
    This provides a more data-driven estimation of costs.

    Args:
        city: The destination city.

    Returns:
        A dictionary with cost of living data, or None on error.
    """
    try:
        # Try to get data from Numbeo API (free tier)
        url = f"https://api.numbeo.com/api/city_cost_estimator"
        params = {
            "api_key": "demo",  # Use demo key for testing
            "city": city,
            "country": "",
            "currency": "USD"
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return None
            
    except Exception as e:
        return None

def get_hotel_prices_api(city, check_in, check_out, adults):
    """
    Fetches real-time hotel prices from a booking API (Booking.com via RapidAPI).

    Args:
        city: The destination city.
        check_in: The check-in date.
        check_out: The check-out date.
        adults: The number of travelers.

    Returns:
        A dictionary with average price and price range, or None on error.
    """
    try:
        # Use RapidAPI to get hotel prices
        url = "https://booking-com.p.rapidapi.com/v1/hotels/search"
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "booking-com.p.rapidapi.com"
        }
        params = {
            "dest_id": city,
            "search_type": "city",
            "arrival_date": check_in,
            "departure_date": check_out,
            "adults": str(adults),
            "room_number": "1",
            "units": "metric",
            "currency": "USD",
            "locale": "en-us"
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            hotels = data.get("result", [])
            if hotels:
                # Calculate average price from top 5 hotels
                prices = []
                for hotel in hotels[:5]:
                    price = hotel.get("price_breakdown", {}).get("gross_price", 0)
                    if price > 0:
                        prices.append(price)
                
                if prices:
                    avg_price = sum(prices) / len(prices)
                    return {
                        "average_price": avg_price,
                        "price_range": f"${min(prices):.0f} - ${max(prices):.0f}",
                        "hotel_count": len(prices)
                    }
        
        return None
        
    except Exception as e:
        return None

def get_food_costs_api(city):
    """
    Estimates daily food costs by analyzing price levels from Google Places API.

    Args:
        city: The destination city.

    Returns:
        A dictionary with average meal and daily food costs, or None on error.
    """
    try:
        # Use Google Places API to get restaurant price levels
        google_hotels = get_google_places(city, "restaurant", max_results=10)
        
        if google_hotels:
            total_cost = 0
            valid_restaurants = 0
            
            for restaurant in google_hotels:
                if restaurant['estimated_cost'] != 'N/A':
                    total_cost += restaurant['estimated_cost']
                    valid_restaurants += 1
            
            if valid_restaurants > 0:
                avg_meal_cost = total_cost / valid_restaurants
                daily_food_cost = avg_meal_cost * 3  # 3 meals per day
                
                return {
                    "average_meal_cost": avg_meal_cost,
                    "daily_food_cost": daily_food_cost,
                    "restaurant_count": valid_restaurants
                }
        
        return None
        
    except Exception as e:
        return None

def get_transport_costs_api(city):
    """
    Provides a basic estimate for daily transportation costs.
    This uses a static dictionary but could be replaced by a live API.

    Args:
        city: The destination city.

    Returns:
        A dictionary with the estimated daily transport cost.
    """
    try:
        # Use Google Maps API to estimate transport costs
        # This is a simplified estimation based on city size and type
        
        # Basic transport cost estimates per day
        transport_estimates = {
            'dubai': 25, 'london': 15, 'new york': 12, 'paris': 8,
            'tokyo': 10, 'singapore': 8, 'bangkok': 5, 'bali': 8,
            'mumbai': 3, 'delhi': 3, 'bangalore': 4, 'chennai': 3,
            'kolkata': 3, 'hyderabad': 4, 'pune': 3
        }
        
        city_lower = city.lower().strip()
        if city_lower in transport_estimates:
            return {
                "daily_transport_cost": transport_estimates[city_lower],
                "includes": "Public transport, occasional taxi/rideshare"
            }
        else:
            return {
                "daily_transport_cost": 8,
                "includes": "Public transport, occasional taxi/rideshare"
            }
            
    except Exception as e:
        return None

def search_with_timeout(func, args, timeout=5):
    """
    Executes a function in a separate thread with a specified timeout.
    This is crucial for preventing long-running network requests from blocking the app.

    Args:
        func: The function to execute.
        args: The arguments to pass to the function.
        timeout: The timeout in seconds.

    Returns:
        The result of the function, or None if it times out.
    """
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        return None
    elif exception[0]:
        # Exception occurred
        raise exception[0]
    else:
        # Function completed successfully
        return result[0]

def search_duckduckgo(query, max_results=5):
    """
    Performs a single search on DuckDuckGo with timeout handling.

    Args:
        query: The search query.
        max_results: The maximum number of results.

    Returns:
        A list of search results, or an empty list on failure or timeout.
    """
    try:
        def _search():
            with DDGS() as ddgs:
                return list(ddgs.text(query, max_results=max_results))
        
        # Use timeout wrapper to prevent hanging
        results = search_with_timeout(_search, (), timeout=5)
        return results if results is not None else []
        
    except Exception as e:
        # Don't print every timeout error to avoid spam
        if "timeout" not in str(e).lower():
            print(f"DuckDuckGo search failed for query '{query}': {str(e)}")
        return []

def search_duckduckgo_with_retry(query, max_results=3, max_retries=1):
    """
    Wraps the DuckDuckGo search with a simple retry mechanism.
    This improves reliability against intermittent network issues.

    Args:
        query: The search query.
        max_results: The maximum number of results.
        max_retries: The number of times to retry on failure.

    Returns:
        A list of search results, or an empty list if all retries fail.
    """
    for attempt in range(max_retries + 1):
        try:
            results = search_duckduckgo(query, max_results)
            if results:
                return results
        except Exception as e:
            if attempt == max_retries:
                # Only log on final failure
                if "timeout" not in str(e).lower():
                    print(f"All retry attempts failed for query '{query}': {str(e)}")
            else:
                # Shorter delay between retries
                time.sleep(0.5)  # Reduced from 1s to 0.5s
    
    return []

def get_real_time_prices_duckduckgo(city, item_type):
    """
    Uses DuckDuckGo to get real-time price estimates for hotels, food, etc.

    Args:
        city: The destination city.
        item_type: The type of item to search for ('hotel', 'food', etc.).

    Returns:
        A list of strings containing price information.
    """
    try:
        queries = {
            'hotel': f"hotel prices {city} 2024",
            'food': f"food cost {city} 2024",
            'transport': f"transport cost {city} 2024",
            'activities': f"tourist prices {city} 2024"
        }
        
        query = queries.get(item_type, f"prices {item_type} {city} 2024")
        results = search_duckduckgo_with_retry(query, max_results=2)  # Reduced from 3 to 2
        
        # Extract price information from search results
        price_info = []
        for result in results:
            title = result.get('title', '')
            body = result.get('body', '')
            price_info.append(f"{title}: {body}")
        
        return price_info
        
    except Exception as e:
        # Silent fail for timeouts
        if "timeout" not in str(e).lower():
            print(f"Error getting {item_type} prices for {city}: {str(e)}")
        return []

def get_local_insights_duckduckgo(city):
    """
    Uses DuckDuckGo to find local insights like best time to visit and travel tips.

    Args:
        city: The destination city.

    Returns:
        A dictionary of insights with queries as keys and results as values.
    """
    try:
        queries = [
            f"best time to visit {city}",
            f"travel tips {city}",
            f"must visit {city}"
        ]
        
        insights = {}
        for query in queries:
            results = search_duckduckgo_with_retry(query, max_results=1)  # Reduced from 2 to 1
            if results:
                insights[query] = [r.get('body', '') for r in results]
        
        return insights
        
    except Exception as e:
        # Silent fail for timeouts
        if "timeout" not in str(e).lower():
            print(f"Error getting local insights for {city}: {str(e)}")
        return {}

def get_weather_insights_duckduckgo(city):
    """
    Uses DuckDuckGo to get high-level weather and climate information.

    Args:
        city: The destination city.

    Returns:
        A dictionary of weather insights.
    """
    try:
        queries = [
            f"weather {city} best time",
            f"climate {city}"
        ]
        
        weather_insights = {}
        for query in queries:
            results = search_duckduckgo_with_retry(query, max_results=1)  # Reduced from 2 to 1
            if results:
                weather_insights[query] = [r.get('body', '') for r in results]
        
        return weather_insights
        
    except Exception as e:
        # Silent fail for timeouts
        if "timeout" not in str(e).lower():
            print(f"Error getting weather insights for {city}: {str(e)}")
        return {}

def get_currency_exchange_duckduckgo(from_currency, to_currency):
    """
    Uses DuckDuckGo to get a real-time currency exchange rate.
    Includes regex parsing to extract the rate from search snippets.

    Args:
        from_currency: The base currency.
        to_currency: The target currency.

    Returns:
        The exchange rate as a float, or None if not found.
    """
    try:
        query = f"exchange rate {from_currency} {to_currency}"
        results = search_duckduckgo_with_retry(query, max_results=1)  # Reduced from 2 to 1
        
        if results:
            # Try to extract exchange rate from search results
            for result in results:
                body = result.get('body', '')
                # Look for patterns like "1 USD = X INR" or "X INR per USD"
                import re
                patterns = [
                    r'1\s*USD\s*=\s*([\d,]+\.?\d*)\s*' + to_currency,
                    r'([\d,]+\.?\d*)\s*' + to_currency + r'\s*per\s*USD',
                    r'([\d,]+\.?\d*)\s*' + to_currency + r'\s*=\s*1\s*USD'
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, body, re.IGNORECASE)
                    if match:
                        rate_str = match.group(1).replace(',', '')
                        try:
                            return float(rate_str)
                        except:
                            continue
        
        return None
        
    except Exception as e:
        # Silent fail for timeouts
        if "timeout" not in str(e).lower():
            print(f"Error getting currency exchange rate: {str(e)}")
        return None

def get_travel_alerts_duckduckgo(city):
    """
    Uses DuckDuckGo to search for any current travel alerts or warnings.

    Args:
        city: The destination city.

    Returns:
        A dictionary of travel alerts.
    """
    try:
        queries = [
            f"travel alerts {city}",
            f"travel warnings {city}"
        ]
        
        alerts = {}
        for query in queries:
            results = search_duckduckgo_with_retry(query, max_results=1)  # Reduced from 2 to 1
            if results:
                alerts[query] = [r.get('body', '') for r in results]
        
        return alerts
        
    except Exception as e:
        # Silent fail for timeouts
        if "timeout" not in str(e).lower():
            print(f"Error getting travel alerts for {city}: {str(e)}")
        return {}

def get_fallback_insights(city):
    """
    Provides a dictionary of static, pre-written insights for popular cities.
    This is used as a fallback when live DuckDuckGo searches fail, ensuring the
    user always gets some useful information.

    Args:
        city: The destination city.

    Returns:
        A dictionary containing fallback insights for the city, or generic tips.
    """
    try:
        # Basic insights for common cities
        fallback_data = {
            'dubai': {
                'local_insights': {
                    'best time to visit': ['Dubai is best visited from November to March when temperatures are pleasant (20-30°C). Avoid summer months (June-September) due to extreme heat.'],
                    'local customs': ['Dress modestly in public areas, especially during Ramadan. Public displays of affection should be avoided.'],
                    'budget travel tips': ['Stay in Deira or Bur Dubai for better value. Use the metro for transportation. Visit during off-peak season for better prices.'],
                    'must visit places': ['Burj Khalifa, Dubai Mall, Palm Jumeirah, Dubai Frame, Dubai Museum, Gold Souk, Dubai Creek.'],
                    'local food recommendations': ['Try Emirati cuisine at Al Ustad Special Kabab, fresh seafood at Bu Qtair, and Arabic coffee at traditional cafes.']
                },
                'weather_insights': {
                    'best weather': ['November to March offers the best weather with temperatures ranging from 20-30°C and minimal rainfall.'],
                    'rainy season': ['Dubai has minimal rainfall, mainly occurring in winter months (December to March).'],
                    'temperature': ['Summer temperatures can reach 45°C, while winter temperatures are pleasant around 20-30°C.']
                },
                'hotel_prices': ['Mid-range hotels in Dubai typically cost $80-150 per night. Luxury hotels can range from $200-500+ per night.'],
                'food_prices': ['Average meal cost: $15-30 for mid-range restaurants, $5-10 for budget options, $50+ for fine dining.']
            },
            'london': {
                'local_insights': {
                    'best time to visit': ['Spring (March-May) and autumn (September-November) offer mild weather and fewer crowds.'],
                    'local customs': ['Queue politely, tip 10-15% in restaurants, and use "please" and "thank you" frequently.'],
                    'budget travel tips': ['Use Oyster cards for transport, visit free museums, and stay in Zone 2-3 for better value.'],
                    'must visit places': ['Big Ben, Buckingham Palace, Tower of London, British Museum, Westminster Abbey, London Eye.'],
                    'local food recommendations': ['Try fish and chips, Sunday roast, afternoon tea, and traditional pub food.']
                },
                'weather_insights': {
                    'best weather': ['May to September offers the warmest weather, though rain is possible year-round.'],
                    'rainy season': ['Rain can occur throughout the year, with October being typically the wettest month.'],
                    'temperature': ['Summer temperatures average 20-25°C, winter temperatures range from 5-10°C.']
                },
                'hotel_prices': ['Mid-range hotels in London typically cost $120-200 per night. Budget options start around $80 per night.'],
                'food_prices': ['Average meal cost: $20-40 for mid-range restaurants, $10-20 for budget options, $60+ for fine dining.']
            },
            'new york': {
                'local_insights': {
                    'best time to visit': ['Spring (April-May) and fall (September-November) offer pleasant weather and fewer crowds.'],
                    'local customs': ['New Yorkers are direct and fast-paced. Tipping 15-20% is expected in restaurants.'],
                    'budget travel tips': ['Use the subway for transportation, visit free attractions like Central Park, and eat at food trucks.'],
                    'must visit places': ['Times Square, Central Park, Statue of Liberty, Empire State Building, Broadway, Metropolitan Museum.'],
                    'local food recommendations': ['Try New York pizza, bagels, hot dogs, and diverse international cuisine in various neighborhoods.']
                },
                'weather_insights': {
                    'best weather': ['Spring and fall offer mild temperatures (15-25°C) and comfortable conditions for sightseeing.'],
                    'rainy season': ['Rain is distributed throughout the year, with spring being slightly wetter.'],
                    'temperature': ['Summer temperatures can reach 30-35°C, winter temperatures range from -5 to 10°C.']
                },
                'hotel_prices': ['Mid-range hotels in NYC typically cost $150-300 per night. Budget options start around $100 per night.'],
                'food_prices': ['Average meal cost: $25-50 for mid-range restaurants, $15-30 for budget options, $80+ for fine dining.']
            },
            'paris': {
                'local_insights': {
                    'best time to visit': ['Spring (April-June) and fall (September-October) offer pleasant weather and beautiful scenery.'],
                    'local customs': ['Learn basic French phrases, greet with "Bonjour", and respect dining customs.'],
                    'budget travel tips': ['Use the metro, visit museums on free days, and stay in less touristy arrondissements.'],
                    'must visit places': ['Eiffel Tower, Louvre Museum, Notre-Dame, Arc de Triomphe, Champs-Élysées, Montmartre.'],
                    'local food recommendations': ['Try croissants, French wine, escargot, coq au vin, and visit local boulangeries.']
                },
                'weather_insights': {
                    'best weather': ['April to October offers the most pleasant weather for exploring the city.'],
                    'rainy season': ['Rain is common throughout the year, with October-November being the wettest months.'],
                    'temperature': ['Summer temperatures average 20-25°C, winter temperatures range from 5-10°C.']
                },
                'hotel_prices': ['Mid-range hotels in Paris typically cost $120-200 per night. Budget options start around $80 per night.'],
                'food_prices': ['Average meal cost: $20-40 for mid-range restaurants, $15-25 for budget options, $60+ for fine dining.']
            },
            'goa': {
                'local_insights': {
                    'best time to visit': ['Goa is best visited from November to March when the weather is pleasant and dry. Avoid monsoon season (June-September) due to heavy rainfall.'],
                    'local customs': ['Dress modestly when visiting religious sites, respect local traditions, and be mindful of the laid-back Goan lifestyle.'],
                    'budget travel tips': ['Stay in North Goa for budget options, use local buses and taxis, and eat at local restaurants for authentic Goan cuisine.'],
                    'must visit places': ['Basilica of Bom Jesus, Aguada Fort, Calangute Beach, Anjuna Flea Market, Dudhsagar Waterfalls, Old Goa churches.'],
                    'local food recommendations': ['Try Goan fish curry, vindaloo, bebinca dessert, feni (local liquor), and fresh seafood at beach shacks.']
                },
                'weather_insights': {
                    'best weather': ['November to March offers the best weather with temperatures ranging from 20-30°C and minimal rainfall.'],
                    'rainy season': ['Monsoon season runs from June to September with heavy rainfall and high humidity.'],
                    'temperature': ['Summer temperatures can reach 35°C, while winter temperatures are pleasant around 20-30°C.']
                },
                'hotel_prices': ['Mid-range hotels in Goa typically cost $40-100 per night. Budget options start around $20 per night.'],
                'food_prices': ['Average meal cost: $5-15 for local restaurants, $2-8 for street food, $20+ for fine dining.']
            }
        }
        
        city_lower = city.lower().strip()
        if city_lower in fallback_data:
            return fallback_data[city_lower]
        else:
            # Generic insights for unknown cities
            return {
                'local_insights': {
                    'general tips': ['Research local customs before visiting, learn basic phrases in the local language, and respect cultural differences.'],
                    'budget advice': ['Look for local markets, use public transportation, and consider staying in less touristy areas for better value.'],
                    'safety tips': ['Keep valuables secure, be aware of your surroundings, and follow local safety guidelines.']
                },
                'weather_insights': {
                    'general weather': ['Check weather forecasts before your trip and pack accordingly for the season.'],
                    'packing tips': ['Pack layers for variable weather, comfortable walking shoes, and weather-appropriate clothing.']
                },
                'hotel_prices': ['Hotel prices vary by location and season. Research accommodation options in advance for best rates.'],
                'food_prices': ['Local food is often more affordable than tourist restaurants. Explore local markets and street food.']
            }
            
    except Exception as e:
        return {}

def enhance_cost_analysis_with_duckduckgo(city, cost_analysis):
    """
    Enhances the existing cost analysis by adding a layer of real-time data
    from DuckDuckGo searches.

    Args:
        city: The destination city.
        cost_analysis: The initial cost analysis dictionary.

    Returns:
        The cost_analysis dictionary, enhanced with DuckDuckGo insights.
    """
    try:
        enhanced_data = cost_analysis.copy()
        
        # Get real-time price information
        hotel_prices = get_real_time_prices_duckduckgo(city, 'hotel')
        food_prices = get_real_time_prices_duckduckgo(city, 'food')
        transport_prices = get_real_time_prices_duckduckgo(city, 'transport')
        activity_prices = get_real_time_prices_duckduckgo(city, 'activities')
        
        # Get local insights
        local_insights = get_local_insights_duckduckgo(city)
        weather_insights = get_weather_insights_duckduckgo(city)
        travel_alerts = get_travel_alerts_duckduckgo(city)
        
        # If DuckDuckGo search failed, use fallback insights
        if not local_insights and not weather_insights:
            fallback_insights = get_fallback_insights(city)
            local_insights = fallback_insights.get('local_insights', {})
            weather_insights = fallback_insights.get('weather_insights', {})
        
        # Add DuckDuckGo data to the analysis
        enhanced_data['duckduckgo_insights'] = {
            'hotel_prices': hotel_prices,
            'food_prices': food_prices,
            'transport_prices': transport_prices,
            'activity_prices': activity_prices,
            'local_insights': local_insights,
            'weather_insights': weather_insights,
            'travel_alerts': travel_alerts
        }
        
        # Update data sources
        if enhanced_data.get('final_estimates'):
            if local_insights or weather_insights:
                enhanced_data['final_estimates']['data_sources'].append('DuckDuckGo real-time search')
            else:
                enhanced_data['final_estimates']['data_sources'].append('Fallback insights')
        
        return enhanced_data
        
    except Exception as e:
        # Return the original cost_analysis if DuckDuckGo enhancement fails
        print(f"DuckDuckGo enhancement failed: {str(e)}")
        return cost_analysis

def get_comprehensive_cost_analysis(city, trip_budget, num_days, num_people, start_date, end_date):
    """
    Orchestrates a multi-source cost analysis.
    It calls multiple APIs (LLM, Booking, Google Places) and aggregates the results
    to produce a more reliable and data-driven cost estimate.

    Args:
        city: The destination city.
        trip_budget: The total trip budget.
        num_days: The number of days.
        num_people: The number of travelers.
        start_date: The trip start date.
        end_date: The trip end date.

    Returns:
        A dictionary containing detailed cost analysis and final estimates.
    """
    try:
        results = {
            "llm_estimates": None,
            "hotel_prices": None,
            "food_costs": None,
            "transport_costs": None,
            "cost_of_living": None,
            "final_estimates": {}
        }
        
        # Get LLM estimates
        results["llm_estimates"] = get_real_time_costs_llm(city, num_people, num_days)
        
        # Get hotel prices
        results["hotel_prices"] = get_hotel_prices_api(city, start_date, end_date, num_people)
        
        # Get food costs
        results["food_costs"] = get_food_costs_api(city)
        
        # Get transport costs
        results["transport_costs"] = get_transport_costs_api(city)
        
        # Get cost of living data
        results["cost_of_living"] = get_cost_of_living_api(city)
        
        # Calculate final estimates
        accommodation_cost = 0
        food_cost = 0
        transport_cost = 0
        
        # Use hotel API data if available
        if results["hotel_prices"]:
            accommodation_cost = results["hotel_prices"]["average_price"]
        elif results["llm_estimates"]:
            accommodation_cost = results["llm_estimates"]["accommodation_per_night"]
        else:
            accommodation_cost = 100 * num_people  # Fallback
        
        # Use food API data if available
        if results["food_costs"]:
            food_cost = results["food_costs"]["daily_food_cost"] * num_people
        elif results["llm_estimates"]:
            food_cost = results["llm_estimates"]["food_per_day"] * num_people
        else:
            food_cost = 30 * num_people  # Fallback
        
        # Use transport API data if available
        if results["transport_costs"]:
            transport_cost = results["transport_costs"]["daily_transport_cost"] * num_people
        elif results["llm_estimates"]:
            transport_cost = results["llm_estimates"]["transport_per_day"] * num_people
        else:
            transport_cost = 8 * num_people  # Fallback
        
        total_daily = accommodation_cost + food_cost + transport_cost
        
        results["final_estimates"] = {
            "accommodation_per_night": round(accommodation_cost, 2),
            "food_per_day": round(food_cost, 2),
            "transport_per_day": round(transport_cost, 2),
            "total_per_day": round(total_daily, 2),
            "total_trip_cost": round(total_daily * num_days, 2),
            "data_sources": []
        }
        
        # Track data sources
        if results["hotel_prices"]:
            results["final_estimates"]["data_sources"].append("Real hotel prices")
        if results["food_costs"]:
            results["final_estimates"]["data_sources"].append("Restaurant price analysis")
        if results["llm_estimates"]:
            results["final_estimates"]["data_sources"].append("LLM research")
        
        # Enhance with DuckDuckGo search data
        results = enhance_cost_analysis_with_duckduckgo(city, results)
        
        return results
        
    except Exception as e:
        return None

def debug_itinerary_format(itinerary_text):
    """
    A helper utility to debug potential issues with the JSON format of an itinerary.
    This is useful for diagnosing problems with the LLM's output.

    Args:
        itinerary_text: The raw itinerary string from the LLM.

    Returns:
        A string indicating the status or error found in the JSON.
    """
    try:
        if not itinerary_text or itinerary_text.strip() == "":
            return "❌ No itinerary text provided"
        
        cleaned_text = itinerary_text.strip()
        
        # Check if it starts with JSON
        if cleaned_text.startswith('{'):
            return "✅ Text starts with JSON object"
        
        # Look for JSON in the text
        start_idx = cleaned_text.find('{')
        end_idx = cleaned_text.rfind('}')
        
        if start_idx == -1:
            return "❌ No opening brace '{' found"
        if end_idx == -1:
            return "❌ No closing brace '}' found"
        if start_idx >= end_idx:
            return "❌ Invalid brace positions"
        
        # Try to extract and parse JSON
        json_part = cleaned_text[start_idx:end_idx+1]
        try:
            json.loads(json_part)
            return f"✅ Valid JSON found at positions {start_idx}-{end_idx}"
        except json.JSONDecodeError as e:
            return f"❌ JSON parsing error: {str(e)}"
            
    except Exception as e:
        return f"❌ Debug error: {str(e)}"

def extract_json_from_formatted_itinerary(formatted_text):
    """
    Extracts a JSON object from a string that might contain other text.
    This is a robust way to handle cases where the LLM includes text
    before or after the JSON object.

    Args:
        formatted_text: The string containing the JSON.

    Returns:
        The extracted JSON string, or None if not found.
    """
    try:
        # Look for JSON in the formatted text
        start_idx = formatted_text.find('{')
        end_idx = formatted_text.rfind('}')
        
        if start_idx != -1 and end_idx != -1:
            json_part = formatted_text[start_idx:end_idx+1]
            # Validate JSON
            json.loads(json_part)
            return json_part
        else:
            return None
    except Exception as e:
        return None

def save_itinerary_to_csv(itinerary_text, city, start_date, end_date):
    """
    Converts a JSON itinerary to a CSV file and saves it locally.
    Includes robust JSON cleaning and validation.

    Args:
        itinerary_text: The JSON string of the itinerary.
        city: The destination city.
        start_date: Trip start date.
        end_date: Trip end date.

    Returns:
        A tuple containing the filename and a status message.
    """
    try:
        # Clean and validate the itinerary text
        if not itinerary_text or itinerary_text.strip() == "":
            return None, "❌ No itinerary data available"
        
        # Try to clean the JSON text (remove any extra text before/after JSON)
        cleaned_text = itinerary_text.strip()
        
        # If it's not valid JSON, try to extract JSON from the text
        if not cleaned_text.startswith('{'):
            # Look for JSON object in the text
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_text = cleaned_text[start_idx:end_idx+1]
            else:
                return None, "❌ No valid JSON found in itinerary"
        
        # Parse the JSON itinerary
        itinerary_data = json.loads(cleaned_text)
        
        # Validate the structure
        if not isinstance(itinerary_data, dict) or 'days' not in itinerary_data:
            return None, "❌ Invalid itinerary structure"
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"itinerary_{city}_{start_date}_to_{end_date}_{timestamp}.csv"
        
        # Write to CSV file
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Day', 'Time', 'Activity', 'Description']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write itinerary data
            for day_data in itinerary_data.get("days", []):
                day_num = day_data.get("day", "N/A")
                
                # Morning activity
                writer.writerow({
                    'Day': f"Day {day_num}",
                    'Time': 'Morning',
                    'Activity': '🌅 Morning Activity',
                    'Description': day_data.get("morning", "No activity planned")
                })
                
                # Lunch
                writer.writerow({
                    'Day': f"Day {day_num}",
                    'Time': 'Lunch',
                    'Activity': '🍽️ Lunch',
                    'Description': day_data.get("lunch", "No lunch planned")
                })
                
                # Afternoon activity
                writer.writerow({
                    'Day': f"Day {day_num}",
                    'Time': 'Afternoon',
                    'Activity': '☀️ Afternoon Activity',
                    'Description': day_data.get("afternoon", "No activity planned")
                })
                
                # Evening activity
                writer.writerow({
                    'Day': f"Day {day_num}",
                    'Time': 'Evening',
                    'Activity': '🌆 Evening Activity',
                    'Description': day_data.get("evening", "No activity planned")
                })
                
                # Dinner
                writer.writerow({
                    'Day': f"Day {day_num}",
                    'Time': 'Dinner',
                    'Activity': '🍴 Dinner',
                    'Description': day_data.get("dinner", "No dinner planned")
                })
                
                # Night activity
                writer.writerow({
                    'Day': f"Day {day_num}",
                    'Time': 'Night',
                    'Activity': '🌙 Night Activity',
                    'Description': day_data.get("night", "No activity planned")
                })
        
        return filename, f"✅ Itinerary saved to {filename}"
        
    except json.JSONDecodeError as e:
        debug_info = debug_itinerary_format(itinerary_text)
        return None, f"❌ JSON parsing error: {str(e)}\n🔍 Debug: {debug_info}"
    except Exception as e:
        return None, f"❌ Error saving itinerary: {str(e)}"

def convert_itinerary_to_csv_data(itinerary_text):
    """
    Converts a JSON itinerary into a list-of-lists format suitable for
    display in the Gradio Dataframe component.

    Args:
        itinerary_text: The JSON string of the itinerary.

    Returns:
        A list of lists representing the CSV data, or None on error.
    """
    try:
        # Clean and validate the itinerary text
        if not itinerary_text or itinerary_text.strip() == "":
            return None
        
        # Try to clean the JSON text (remove any extra text before/after JSON)
        cleaned_text = itinerary_text.strip()
        
        # If it's not valid JSON, try to extract JSON from the text
        if not cleaned_text.startswith('{'):
            # Look for JSON object in the text
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_text = cleaned_text[start_idx:end_idx+1]
            else:
                return None
        
        # Parse the JSON itinerary
        itinerary_data = json.loads(cleaned_text)
        
        # Validate the structure
        if not isinstance(itinerary_data, dict) or 'days' not in itinerary_data:
            return None
        
        # Create CSV data as list of lists
        csv_data = []
        
        # Add itinerary data
        for day_data in itinerary_data.get("days", []):
            day_num = day_data.get("day", "N/A")
            
            # Morning activity
            morning = day_data.get("morning", "No activity planned")
            csv_data.append([f"Day {day_num}", "🌅 Morning", "Morning Activity", morning])
            
            # Lunch
            lunch = day_data.get("lunch", "No lunch planned")
            csv_data.append([f"Day {day_num}", "🍽️ Lunch", "Lunch", lunch])
            
            # Afternoon activity
            afternoon = day_data.get("afternoon", "No activity planned")
            csv_data.append([f"Day {day_num}", "☀️ Afternoon", "Afternoon Activity", afternoon])
            
            # Evening activity
            evening = day_data.get("evening", "No activity planned")
            csv_data.append([f"Day {day_num}", "🌆 Evening", "Evening Activity", evening])
            
            # Dinner
            dinner = day_data.get("dinner", "No dinner planned")
            csv_data.append([f"Day {day_num}", "🍴 Dinner", "Dinner", dinner])
            
            # Night activity
            night = day_data.get("night", "No activity planned")
            csv_data.append([f"Day {day_num}", "🌙 Night", "Night Activity", night])
        
        return csv_data
        
    except json.JSONDecodeError as e:
        return None
    except Exception as e:
        return None

def convert_itinerary_to_csv(itinerary_text, city, start_date, end_date):
    """
    Converts a JSON itinerary into a CSV formatted string.
    This is used for creating a downloadable file in Gradio without
    saving to disk on the server side first.

    Args:
        itinerary_text: The JSON string of the itinerary.
        city: The destination city.
        start_date: Trip start date.
        end_date: Trip end date.

    Returns:
        A string containing the CSV data, or an error message.
    """
    try:
        # Clean and validate the itinerary text
        if not itinerary_text or itinerary_text.strip() == "":
            return "Error: No itinerary data available"
        
        # Try to clean the JSON text (remove any extra text before/after JSON)
        cleaned_text = itinerary_text.strip()
        
        # If it's not valid JSON, try to extract JSON from the text
        if not cleaned_text.startswith('{'):
            # Look for JSON object in the text
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                cleaned_text = cleaned_text[start_idx:end_idx+1]
            else:
                return "Error: No valid JSON found in itinerary"
        
        # Parse the JSON itinerary
        itinerary_data = json.loads(cleaned_text)
        
        # Validate the structure
        if not isinstance(itinerary_data, dict) or 'days' not in itinerary_data:
            return "Error: Invalid itinerary structure"
        
        # Create CSV content as string
        csv_content = "Day,Time,Activity,Description\n"
        
        # Add itinerary data
        for day_data in itinerary_data.get("days", []):
            day_num = day_data.get("day", "N/A")
            
            # Morning activity
            morning = day_data.get("morning", "No activity planned").replace('"', '""')
            csv_content += f'Day {day_num},🌅 Morning,Morning Activity,"{morning}"\n'
            
            # Lunch
            lunch = day_data.get("lunch", "No lunch planned").replace('"', '""')
            csv_content += f'Day {day_num},🍽️ Lunch,Lunch,"{lunch}"\n'
            
            # Afternoon activity
            afternoon = day_data.get("afternoon", "No activity planned").replace('"', '""')
            csv_content += f'Day {day_num},☀️ Afternoon,Afternoon Activity,"{afternoon}"\n'
            
            # Evening activity
            evening = day_data.get("evening", "No activity planned").replace('"', '""')
            csv_content += f'Day {day_num},🌆 Evening,Evening Activity,"{evening}"\n'
            
            # Dinner
            dinner = day_data.get("dinner", "No dinner planned").replace('"', '""')
            csv_content += f'Day {day_num},🍴 Dinner,Dinner,"{dinner}"\n'
            
            # Night activity
            night = day_data.get("night", "No activity planned").replace('"', '""')
            csv_content += f'Day {day_num},🌙 Night,Night Activity,"{night}"\n'
        
        return csv_content
        
    except json.JSONDecodeError as e:
        return f"Error: JSON parsing error - {str(e)}"
    except Exception as e:
        return f"Error converting itinerary: {str(e)}"

def export_to_markdown(itinerary_text, city, start_date, end_date, weather_info, budget_info, hotel_info, google_info, duckduckgo_info):
    """
    Exports the entire travel plan into a single, well-formatted Markdown file.
    This creates a professional, shareable document for the user.

    Args:
        itinerary_text: The JSON itinerary.
        city: Destination city.
        start_date: Trip start date.
        end_date: Trip end date.
        weather_info: Formatted weather string.
        budget_info: Formatted budget string.
        hotel_info: Formatted hotel string.
        google_info: Formatted Google Places string.
        duckduckgo_info: Formatted DuckDuckGo insights string.

    Returns:
        A tuple containing the filename and a status message.
    """
    try:
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"travel_plan_{city}_{start_date}_to_{end_date}_{timestamp}.md"
        
        # Start building markdown content
        markdown_content = f"""# 🌏 Travel Plan: {city}

**Trip Dates:** {start_date} to {end_date}  
**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## 🌤️ Weather Information

{weather_info if weather_info else "Weather information not available"}

---

## 💰 Budget Summary

{budget_info if budget_info else "Budget information not available"}

---

## 🏨 Hotel Options

{hotel_info if hotel_info else "Hotel information not available"}

---

## 🏨🍽️ Google Places Recommendations

{google_info if google_info else "Google Places information not available"}

---

## 🔍 Real-Time Insights

{duckduckgo_info if duckduckgo_info else "Real-time insights not available"}

---

## 📖 AI-Powered Itinerary

"""
        
        # Add itinerary in a formatted way
        if itinerary_text:
            try:
                # Try to parse and format JSON itinerary
                itinerary_data = json.loads(itinerary_text.strip())
                if isinstance(itinerary_data, dict) and 'days' in itinerary_data:
                    for day_data in itinerary_data.get("days", []):
                        day_num = day_data.get("day", "N/A")
                        markdown_content += f"\n### Day {day_num}\n\n"
                        
                        # Morning
                        morning = day_data.get("morning", "No activity planned")
                        markdown_content += f"**🌅 Morning:** {morning}\n\n"
                        
                        # Lunch
                        lunch = day_data.get("lunch", "No lunch planned")
                        markdown_content += f"**🍽️ Lunch:** {lunch}\n\n"
                        
                        # Afternoon
                        afternoon = day_data.get("afternoon", "No activity planned")
                        markdown_content += f"**☀️ Afternoon:** {afternoon}\n\n"
                        
                        # Evening
                        evening = day_data.get("evening", "No activity planned")
                        markdown_content += f"**🌆 Evening:** {evening}\n\n"
                        
                        # Dinner
                        dinner = day_data.get("dinner", "No dinner planned")
                        markdown_content += f"**🍴 Dinner:** {dinner}\n\n"
                        
                        # Night
                        night = day_data.get("night", "No activity planned")
                        markdown_content += f"**🌙 Night:** {night}\n\n"
                        
                        markdown_content += "---\n\n"
                else:
                    # If JSON parsing fails, add raw text
                    markdown_content += f"```json\n{itinerary_text}\n```\n\n"
            except json.JSONDecodeError:
                # If JSON parsing fails, add raw text
                markdown_content += f"```json\n{itinerary_text}\n```\n\n"
        else:
            markdown_content += "Itinerary not available\n\n"
        
        # Add footer
        markdown_content += """---

## 📝 Notes

- This travel plan was generated by an AI-powered travel assistant
- All prices and availability should be verified before booking
- Weather forecasts may change, so check closer to your travel dates
- Consider local customs and travel advisories for your destination

---

*Generated with 🌏 Enhanced AI Travel Agent & Real-Time Hotel Planner*
"""
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        return filename, f"✅ Travel plan exported to {filename}"
        
    except Exception as e:
        return None, f"❌ Error exporting to markdown: {str(e)}"

def test_duckduckgo_connection():
    """
    Tests the connection to DuckDuckGo to provide a status indicator in the UI.
    This helps users understand if real-time features are currently active.

    Returns:
        A tuple containing a boolean status and a descriptive message.
    """
    try:
        # Use a very short timeout for testing
        def _test_search():
            return search_duckduckgo_with_retry("test query", max_results=1, max_retries=0)
        
        test_results = search_with_timeout(_test_search, (), timeout=3)
        
        if test_results is None:
            return False, "DuckDuckGo search timed out (network connectivity issue)"
        elif test_results:
            return True, "DuckDuckGo search is working"
        else:
            return False, "DuckDuckGo search returned no results (may be due to network timeouts)"
    except Exception as e:
        if "timeout" in str(e).lower():
            return False, "DuckDuckGo search timed out (network connectivity issue)"
        else:
            return False, f"DuckDuckGo search failed: {str(e)}"

# ============== ENHANCED TRAVEL PLANNER CLASS ==============

class EnhancedTravelPlanner:
    """
    An advanced travel planner that uses a LangGraph agent workflow.
    This class orchestrates multiple search APIs and LLM calls to generate
    a more detailed and context-aware travel plan.
    """
    
    def __init__(self, config: Config):
        """
        Initializes the EnhancedTravelPlanner.

        Args:
            config: A Config object containing all necessary API keys.
        """
        self.config = config
        
        # Initialize various search tools with real-time capabilities
        self.search_tool = DuckDuckGoSearchRun()
        
        # Initialize Google Places for real-time location data, if API key is available
        try:
            if config.google_api_key:
                places_wrapper = GooglePlacesAPIWrapper(api_key=config.google_api_key)
                self.places_tool = GooglePlacesTool(api_wrapper=places_wrapper)
            else:
                self.places_tool = None
        except Exception:
            self.places_tool = None
            
        # Initialize SerpAPI for real-time Google search results, if API key is available
        try:
            if config.serpapi_key:
                self.serp_search = SerpAPIWrapper(api_key=config.serpapi_key)
            else:
                self.serp_search = None
        except Exception:
            self.serp_search = None
            
        # Initialize Google Serper for real-time search, if API key is available
        try:
            if config.serper_api_key:
                self.serper_search = GoogleSerperAPIWrapper(api_key=config.serper_api_key)
            else:
                self.serper_search = None
        except Exception:
            self.serper_search = None
        
        # Initialize the Large Language Model (LLM)
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=config.openai_api_key,
        )
        
        # Setup the suite of tools the agent can use
        self.tools = self._setup_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build the graph, which represents the agent's reasoning flow
        self.graph = self._build_graph()
    
    def _setup_tools(self) -> List:
        """
        Initializes and defines the suite of tools available to the LangGraph agent.
        Each tool is a specific capability, like searching for hotels or getting weather.
        It uses multiple search providers (Google Places, SerpAPI, etc.) with fallbacks.
        """
        
        @tool
        def search_attractions(city: str) -> str:
            """Search for top attractions in a city using a cascade of real-time data sources."""
            query = f"top attractions activities things to do in {city}"
            
            # Try Google Places first for structured, real-time data
            if self.places_tool:
                try:
                    places_result = self.places_tool.invoke(f"tourist attractions in {city}")
                    if places_result and len(places_result) > 50:
                        return f"Real-time attractions data: {places_result}"
                except Exception:
                    pass
            
            # Try SerpAPI for fresh Google results
            if self.serp_search:
                try:
                    serp_result = self.serp_search.invoke(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Latest search results: {serp_result}"
                except Exception:
                    pass
            
            # Try Google Serper
            if self.serper_search:
                try:
                    serper_result = self.serper_search.invoke(query)
                    if serper_result and len(serper_result) > 50:
                        return f"Latest hotel availability: {serper_result}"
                except Exception:
                    pass
            
            # Fallback to DuckDuckGo
            return self.search_tool.invoke(query)
        
        @tool
        def search_restaurants(city: str) -> str:
            """Search for restaurants in a city using a cascade of real-time data sources."""
            query = f"best restaurants food places to eat in {city}"
            
            # Try Google Places first for real-time restaurant data
            if self.places_tool:
                try:
                    places_result = self.places_tool.invoke(f"restaurants in {city}")
                    if places_result and len(places_result) > 50:
                        return f"Real-time restaurant data: {places_result}"
                except Exception:
                    pass
            
            # Try SerpAPI for current results
            if self.serp_search:
                try:
                    serp_result = self.serp_search.invoke(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Latest restaurant results: {serp_result}"
                except Exception:
                    pass
            
            # Fallback to DuckDuckGo
            return self.search_tool.invoke(query)
        
        @tool
        def search_hotels(city: str, budget_range: str = "mid-range") -> str:
            """Search for hotels in a city with budget range using a cascade of real-time data sources."""
            query = f"{budget_range} hotels accommodation {city} price per night booking availability"
            
            # Try SerpAPI first for real-time hotel prices and availability
            if self.serp_search:
                try:
                    serp_result = self.serp_search.run(query)
                    if serp_result and len(serp_result) > 50:
                        return f"Real-time hotel data: {serp_result}"
                except Exception:
                    pass
            
            # Try Google Places for hotel information
            if self.places_tool:
                try:
                    places_result = self.places_tool.invoke(f"hotels in {city}")
                    if places_result and len(places_result) > 50:
                        return f"Current hotel listings: {places_result}"
                except Exception:
                    pass
            
            # Try Google Serper
            if self.serper_search:
                try:
                    serper_result = self.serper_search.run(query)
                    if serper_result and len(serper_result) > 50:
                        return f"Latest hotel availability: {serper_result}"
                except Exception:
                    pass
            
            # Fallback to DuckDuckGo
            return self.search_tool.invoke(query)
        
        @tool
        def get_weather_info(city: str) -> str:
            """Get current weather and a 5-day forecast for a city."""
            try:
                current_weather = get_current_weather(city)
                forecast = get_weather_forecast(city)
                
                weather_info = f"Current weather in {city}: {current_weather}\n\n"
                if forecast:
                    weather_info += "Forecast:\n"
                    for date, info in list(forecast.items())[:5]:  # Show next 5 days
                        weather_info += f"{date}: {info}\n"
                
                return weather_info
            except Exception as e:
                return f"Could not fetch weather for {city}: {str(e)}"
        
        @tool
        def get_exchange_rate_info(from_currency: str, to_currency: str) -> str:
            """Get the exchange rate between two currencies."""
            try:
                rate = get_exchange_rate(from_currency, to_currency)
                if rate:
                    return f"Exchange rate: 1 {from_currency} = {rate:.4f} {to_currency}"
                else:
                    return f"Could not fetch exchange rate for {from_currency} to {to_currency}"
            except Exception as e:
                return f"Error getting exchange rate: {str(e)}"
        
        @tool
        def calculate_budget_breakdown(total_budget: float, num_days: int, num_people: int) -> str:
            """Calculate a detailed budget breakdown for a trip based on user inputs."""
            try:
                accommodation_budget = total_budget * 0.5
                food_budget = total_budget * 0.2
                travel_budget = total_budget * 0.1
                leisure_budget = total_budget * 0.2
                
                per_night_budget = accommodation_budget / num_days
                daily_budget = total_budget / num_days
                
                breakdown = f"""Budget Breakdown for {num_days} days, {num_people} person(s):
Total Budget: ${total_budget:,.2f}
Daily Budget: ${daily_budget:,.2f}

Accommodation: ${accommodation_budget:,.2f} (50%)
Food: ${food_budget:,.2f} (20%)
Travel: ${travel_budget:,.2f} (10%)
Leisure: ${leisure_budget:,.2f} (20%)

Per Night Budget: ${per_night_budget:,.2f}"""
                
                return breakdown
            except Exception as e:
                return f"Error calculating budget: {str(e)}"
        
        return [
            search_attractions, search_restaurants, search_hotels,
            get_weather_info, get_exchange_rate_info, calculate_budget_breakdown
        ]
    
    def _build_graph(self) -> StateGraph:
        """
        Builds the LangGraph state machine (workflow).
        This defines the agent's logic: how it moves between thinking (agent node)
        and acting (tool node) until it produces a complete answer.
        """
        
        def agent_function(state: MessagesState):
            """
            The main "brain" of the agent. It decides which tool to use based on the user's
            request and the current state of the conversation.
            """
            user_question = state["messages"]
            system_prompt = SystemMessage(
                content="""You are a helpful AI Travel Agent and Expense Planner. 
                You help users plan trips to any city worldwide with real-time data.
                
                IMPORTANT: Always provide COMPLETE and DETAILED travel plans. Never say "I'll prepare" or "hold on". 
                Give full information immediately including:
                - Complete day-by-day itinerary
                - Specific attractions with details
                - Restaurant recommendations with prices
                - Detailed cost breakdown
                - Transportation information
                - Weather details
                
                Use the available tools to gather real-time information and make accurate calculations.
                Provide everything in one comprehensive response formatted in clean Markdown.
                """
            )
            input_question = [system_prompt] + user_question
            response = self.llm_with_tools.invoke(input_question)
            return {"messages": [response]}
        
        def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
            """
            The conditional logic that determines if the agent should continue working.
            It checks if the agent has called a tool or if the response is still
            incomplete, preventing premature termination.
            """
            last_message = state["messages"][-1]
            
            # If the agent decides to use a tool, we should continue.
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            
            # Check for various indicators that the response is incomplete.
            content = last_message.content.lower()
            
            # If the agent gives a placeholder response, it needs to continue working.
            incomplete_phrases = [
                "let me search",
                "i'll look up",
                "please hold on",
                "i'll prepare",
                "let me gather",
                "i need to check",
                "searching for",
                "looking up"
            ]
            
            # If response contains incomplete phrases, continue
            if any(phrase in content for phrase in incomplete_phrases):
                return "tools"
            
            # Check if response is too short (likely incomplete)
            if len(last_message.content) < 300:
                return "tools"
            
            # Check if we have essential travel info
            essential_keywords = ["hotel", "attraction", "cost", "weather", "itinerary", "restaurant", "budget"]
            has_essential_info = sum(1 for keyword in essential_keywords if keyword in content) >= 4
            
            if not has_essential_info:
                return "tools"
            
            # Check for complete structure indicators
            structure_indicators = ["day 1", "morning", "afternoon", "evening", "dinner", "budget", "cost"]
            has_structure = sum(1 for indicator in structure_indicators if indicator in content) >= 3
            
            if not has_structure:
                return "tools"
            
            # If none of the above conditions are met, the response is likely complete.
            return "__end__"
        
        # Create the state graph instance
        workflow = StateGraph(MessagesState)
        workflow.add_node("agent", agent_function)
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges with better control
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        
        # Compile with recursion limit to prevent infinite loops
        return workflow.compile()
    
    def plan_trip_enhanced(self, user_input: str, max_iterations: int = 15) -> str:
        """Enhanced trip planning using LangGraph workflow"""
        messages = [HumanMessage(content=user_input)]
        
        # Add iteration counter to prevent infinite loops with higher limit
        config = {"recursion_limit": max_iterations}
        
        try:
            response = self.graph.invoke({"messages": messages}, config=config)
            final_response = response["messages"][-1].content
            
            # Final check - if still incomplete, force a summary
            if len(final_response) < 500:
                summary_prompt = f"""
                Based on all the information gathered, provide a COMPLETE travel summary now. 
                Don't use tools anymore. Use the information you have to create a comprehensive plan.
                Format your response in clean Markdown with proper headers, lists, and formatting.
                Original request: {user_input}
                """
                
                summary_messages = response["messages"] + [HumanMessage(content=summary_prompt)]
                final_response_obj = self.llm_with_tools.invoke(summary_messages)
                return final_response_obj.content
            
            return final_response
            
        except Exception as e:
            print(f"Workflow error: {e}")
            # Fallback - direct LLM call
            return self._fallback_planning(user_input)
    
    def _fallback_planning(self, user_input: str) -> str:
        """Fallback method if workflow fails"""
        fallback_prompt = f"""
        Create a complete travel plan for: {user_input}
        
        Provide a comprehensive response including:
        - Daily itinerary
        - Top attractions
        - Restaurant recommendations  
        - Cost estimates
        - Weather information
        - Transportation details
        
        Format your response in clean Markdown with proper headers, lists, and formatting.
        Use your knowledge to provide helpful estimates even without real-time data.
        """
        
        system_prompt = SystemMessage(
            content="You are a helpful AI Travel Agent. Provide comprehensive travel plans with detailed information."
        )
        messages = [system_prompt, HumanMessage(content=fallback_prompt)]
        response = self.llm_with_tools.invoke(messages)
        return response.content

# ============== GRADIO INTERFACE ==============
# This section defines the user interface of the application using the Gradio library.

def create_interface():
    """
    Creates and configures the entire Gradio web interface, including all components,
    layouts, and event-handling functions.
    """
    with gr.Blocks(title="🌏 Enhanced AI Travel Agent & Real-Time Hotel Planner", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🌏 Enhanced AI Travel Agent & Real-Time Hotel Planner")
        gr.Markdown("Plan your perfect trip with AI-powered recommendations, real-time weather, and hotel options!")
        
        # Initialize the enhanced travel planner class, which contains the LangGraph agent
        config = Config()
        enhanced_planner = EnhancedTravelPlanner(config)
        
        with gr.Row():
            with gr.Column():
                # --- User Input Components ---
                gr.Markdown("### 📝 Your Trip Details")
                city = gr.Textbox(label="Destination City", placeholder="e.g., Dubai, London, New York")
                with gr.Row():
                    start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", placeholder="e.g., 2024-06-15")
                    end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", placeholder="e.g., 2024-06-20")
                num_people = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Number of Travelers")
                interests = gr.Textbox(label="Interests", placeholder="e.g., history, beaches, museums, food, adventure")
                with gr.Row():
                    trip_budget = gr.Number(label="Total Trip Budget (USD)", value=1000.0, minimum=100.0)
                    currency_code = gr.Textbox(label="Your Currency Code", value="INR", placeholder="e.g., INR, EUR, GBP")
                
                # --- Advanced Options & Toggles ---
                gr.Markdown("### ✨ Advanced Options")
                with gr.Row():
                    use_enhanced_planner = gr.Checkbox(label="🚀 Use Enhanced LangGraph Workflow", value=False, info="Enable advanced agent workflow with multiple search APIs")
                    use_markdown_export = gr.Checkbox(label="📄 Enable Markdown Export", value=True, info="Generate comprehensive markdown documentation")
                    use_real_time_data = gr.Checkbox(label="🔍 Real-Time Data Sources", value=True, info="Use DuckDuckGo and multiple APIs for live data")
                
                with gr.Row():
                    planning_style = gr.Dropdown(
                        choices=["Balanced", "Budget-Friendly", "Luxury", "Adventure", "Cultural", "Relaxation"],
                        value="Balanced",
                        label="🎯 Planning Style",
                        info="Choose your preferred travel style"
                    )
                    trip_pace = gr.Dropdown(
                        choices=["Relaxed", "Moderate", "Fast-paced"],
                        value="Moderate",
                        label="⏱️ Trip Pace",
                        info="How active do you want your trip to be?"
                    )
                
                plan_button = gr.Button("🚀 Plan My Trip", variant="primary", size="lg")
            
            with gr.Column():
                # --- Output Display Components ---
                gr.Markdown("### 📋 Trip Summary")
                with gr.Tabs():
                    with gr.TabItem("📊 Overview"):
                        with gr.Row():
                            weather_output = gr.Textbox(label="🌤 Weather Forecast", lines=8, interactive=False)
                            budget_output = gr.Textbox(label="💰 Budget Summary", lines=22, interactive=False)
                        duckduckgo_output = gr.Textbox(label="🔍 Real-Time Insights", lines=20, interactive=False)
                    
                    with gr.TabItem("🏨 Hotels & Restaurants"):
                        hotels_output = gr.Textbox(label="🏨 Hotel Options (TripAdvisor)", lines=10, interactive=False)
                        google_output = gr.Textbox(label="🏨🍽️ Google Places", lines=15, interactive=False)
                        
                    with gr.TabItem("📖 Itinerary"):
                        itinerary_output = gr.Textbox(label="📖 AI-Powered Itinerary (JSON Format)", lines=25, interactive=False)
                        csv_table_output = gr.Dataframe(
                            headers=["Day", "Time", "Activity", "Description"],
                            label="📊 Itinerary Table View",
                            visible=True,
                            interactive=False,
                            column_widths=["10%", "15%", "20%", "55%"]
                        )
                
                # Enhanced output for LangGraph workflow - initially hidden
                enhanced_output = gr.Markdown(label="🤖 Enhanced AI Analysis & Additional Insights", visible=False)
                
                # --- Export Section ---
                gr.Markdown("### 📤 Export Plan")
                with gr.Row():
                    csv_download_btn = gr.Button("📥 Download Itinerary as CSV", variant="secondary")
                    markdown_download_btn = gr.Button("📄 Export to Markdown", variant="secondary")
                save_status = gr.Textbox(label="💾 Save Status", lines=2, interactive=False, placeholder="Download status will appear here...")
                
                # Hidden file components for triggering downloads
                csv_file_output = gr.File(label="📄 Download CSV File", visible=False)
                markdown_file_output = gr.File(label="📄 Download Markdown File", visible=False)
        
        # --- Core Application Logic ---
        
        def plan_trip_enhanced(city, start_date, end_date, num_people, interests, trip_budget, currency_code, use_enhanced, use_markdown, use_real_time, planning_style, trip_pace):
            """
            This is the main function connected to the 'Plan My Trip' button.
            It orchestrates the trip planning process based on user inputs and toggles,
            including the optional enhanced LangGraph workflow.
            """
            # Always generate the regular output first as a baseline
            weather, budget, hotels, google, itinerary, duckduckgo = plan_trip(city, start_date, end_date, num_people, interests, trip_budget, currency_code)
            
            # Initialize the enhanced result to be empty
            enhanced_result = ""
            
            # If the user has enabled the enhanced workflow, run the LangGraph agent
            if use_enhanced:
                user_input = f"Plan a {num_people}-person {planning_style.lower()} trip to {city} from {start_date} to {end_date} with {trip_pace.lower()} pace. Interests: {interests}. Budget: ${trip_budget} USD. Convert budget to {currency_code}. Provide additional insights and recommendations."
                try:
                    enhanced_result = enhanced_planner.plan_trip_enhanced(user_input)
                except Exception as e:
                    # Provide a graceful error message if the enhanced workflow fails
                    enhanced_result = f"🤖 Enhanced analysis temporarily unavailable: {str(e)}\n\n💡 The regular travel plan is still fully functional!"
            else:
                # When enhanced mode is off, provide a simple informational message
                enhanced_result = f"🤖 **Enhanced AI Analysis Disabled**\n\n**Planning Style:** {planning_style}\n**Trip Pace:** {trip_pace}\n\nEnable the 'Use Enhanced LangGraph Workflow' option for advanced AI-powered insights and recommendations."
            
            # The function must always return 7 values to match the number of output components in the UI
            return weather, budget, hotels, google, itinerary, enhanced_result, duckduckgo
        
        # Connect the button click event to the main planning function
        plan_button.click(
            fn=plan_trip_enhanced,
            inputs=[city, start_date, end_date, num_people, interests, trip_budget, currency_code, use_enhanced_planner, use_markdown_export, use_real_time_data, planning_style, trip_pace],
            outputs=[weather_output, budget_output, hotels_output, google_output, itinerary_output, enhanced_output, duckduckgo_output]
        )
        
        # Toggle enhanced output visibility
        def toggle_enhanced_output(use_enhanced):
            if use_enhanced:
                return gr.update(visible=True, value="🤖 **Enhanced AI Analysis Enabled**\n\nWhen you plan your trip, you'll see both the regular output and additional AI-powered insights here!")
            else:
                return gr.update(visible=False, value="")
        
        use_enhanced_planner.change(
            fn=toggle_enhanced_output,
            inputs=[use_enhanced_planner],
            outputs=[enhanced_output]
        )
        
        # Connect CSV download button
        def download_and_display_csv(itinerary_text, city, start_date, end_date):
            """Download CSV file and display table"""
            if not itinerary_text or itinerary_text.strip() == "":
                return "❌ No itinerary available. Please plan a trip first.", None, None
            
            try:
                # Extract JSON from formatted itinerary
                json_text = extract_json_from_formatted_itinerary(itinerary_text)
                if not json_text:
                    return "❌ Could not extract JSON from formatted itinerary for CSV export.", None, None
                
                # Debug the itinerary format
                debug_info = "🔍 Extracted JSON from formatted itinerary"
                
                # Save to file using the extracted JSON
                filename, status = save_itinerary_to_csv(json_text, city, start_date, end_date)
                if filename:
                    # Also create CSV data for table display
                    csv_data = convert_itinerary_to_csv_data(json_text)
                    if csv_data:
                        return f"✅ Itinerary saved as: {filename}\n📁 File location: {os.path.abspath(filename)}\n{debug_info}", filename, csv_data
                    else:
                        return f"✅ Itinerary saved as: {filename}\n📁 File location: {os.path.abspath(filename)}\n{debug_info}\n⚠️ Could not display table", filename, None
                else:
                    return f"{status}\n{debug_info}", None, None
            except Exception as e:
                return f"❌ Error: {str(e)}", None, None
        
        def create_csv_file(itinerary_text, city, start_date, end_date):
            """Create CSV file for direct download"""
            if not itinerary_text or itinerary_text.strip() == "":
                return None
            
            try:
                # Extract JSON from formatted itinerary
                json_text = extract_json_from_formatted_itinerary(itinerary_text)
                if not json_text:
                    return None
                
                # Convert to CSV content
                csv_content = convert_itinerary_to_csv(json_text, city, start_date, end_date)
                
                # Check if conversion was successful
                if csv_content.startswith("Error:"):
                    return None
                
                # Create filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"itinerary_{city}_{start_date}_to_{end_date}_{timestamp}.csv"
                
                # Write to temporary file
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    f.write(csv_content)
                
                return filename
            except Exception as e:
                return None
        
        csv_download_btn.click(
            fn=download_and_display_csv,
            inputs=[itinerary_output, city, start_date, end_date],
            outputs=[save_status, csv_file_output, csv_table_output]
        )
        
        # Markdown export functionality
        def download_markdown(weather_info, budget_info, hotel_info, google_info, itinerary_text, duckduckgo_info, city, start_date, end_date):
            """Download comprehensive travel plan as Markdown"""
            if not city or not start_date or not end_date:
                return "❌ Please provide city and travel dates first.", None
            
            try:
                filename, status = export_to_markdown(
                    itinerary_text, city, start_date, end_date,
                    weather_info, budget_info, hotel_info, google_info, duckduckgo_info
                )
                if filename:
                    return f"{status}\n📁 File location: {os.path.abspath(filename)}", filename
                else:
                    return status, None
            except Exception as e:
                return f"❌ Error: {str(e)}", None
        
        markdown_download_btn.click(
            fn=download_markdown,
            inputs=[weather_output, budget_output, hotels_output, google_output, itinerary_output, duckduckgo_output, city, start_date, end_date],
            outputs=[save_status, markdown_file_output]
        )
        
        # Add some helpful information
        gr.Markdown("""
        ### 💡 Tips for Best Results:
        - Use specific city names (e.g., "Dubai" not "UAE")
        - Set realistic budgets for better hotel recommendations
        - Include specific interests for personalized itineraries
        - Check weather forecasts to plan activities accordingly
        - Enable Enhanced LangGraph Workflow for advanced AI analysis
        
        ### 🔧 Features:
        - Real-time weather data and forecasts
        - Budget planning with currency conversion
        - TripAdvisor hotel recommendations
        - Google Places popular hotels and restaurants
        - AI-generated daily itineraries
        - **NEW**: DuckDuckGo real-time search for current prices, travel alerts, and local insights
        - **NEW**: Multi-source cost analysis with real-time data validation
        - **NEW**: LangGraph workflow with multiple search APIs (SerpAPI, Google Serper)
        - **NEW**: Markdown export with comprehensive travel documentation
        - **NEW**: Enhanced agent workflow with intelligent tool orchestration
        
        ### 🔍 DuckDuckGo Status:
        """)
        
        # Test DuckDuckGo connection and show status
        try:
            duckduckgo_working, duckduckgo_status = test_duckduckgo_connection()
            status_emoji = "✅" if duckduckgo_working else "⚠️"
            gr.Markdown(f"{status_emoji} **DuckDuckGo Search**: {duckduckgo_status}")
        except:
            gr.Markdown("⚠️ **DuckDuckGo Search**: Status unknown")
    
    return demo

# Launch the app
if __name__ == "__main__":
    print("🚀 Starting Enhanced AI Travel Agent...")
    print("📱 Opening web interface...")
    demo = create_interface()
    demo.launch(share=False, debug=True, server_name="127.0.0.1", server_port=7860) 