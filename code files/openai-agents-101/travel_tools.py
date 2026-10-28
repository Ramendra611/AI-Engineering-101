"""Mock travel tools shared by the labs.

Real agents call real services (weather APIs, airline systems, payment
gateways). These functions return fixed, made-up data instead, so every lab
runs offline, costs nothing beyond the model calls, and behaves the same way
every time. Only the model's wording varies between runs.
"""

import uuid

WEATHER = {
    "goa": {"temp_c": 31, "condition": "sunny", "humidity_pct": 70},
    "mumbai": {"temp_c": 29, "condition": "humid and cloudy", "humidity_pct": 82},
    "delhi": {"temp_c": 24, "condition": "hazy", "humidity_pct": 45},
    "jaipur": {"temp_c": 27, "condition": "clear", "humidity_pct": 30},
    "manali": {"temp_c": 9, "condition": "cold and clear", "humidity_pct": 40},
}

FLIGHTS = [  # fictional flights, prices in Indian rupees (INR)
    {"id": "FL-101", "origin": "Mumbai", "destination": "Goa",
        "departs": "07:10", "price_inr": 4200},
    {"id": "FL-202", "origin": "Mumbai", "destination": "Goa",
        "departs": "13:45", "price_inr": 3500},
    {"id": "FL-303", "origin": "Delhi", "destination": "Goa",
        "departs": "09:30", "price_inr": 6100},
    {"id": "FL-404", "origin": "Delhi", "destination": "Jaipur",
        "departs": "18:00", "price_inr": 2900},
    {"id": "FL-505", "origin": "Mumbai", "destination": "Jaipur",
        "departs": "11:15", "price_inr": 5200},
]

RATES_TO_INR = {"INR": 1.0, "USD": 88.0, "EUR": 103.0,
                "GBP": 118.0}  # fixed demo rates, not live


def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: City name, for example "Goa".
    """
    data = WEATHER.get(city.strip().lower())
    if data is None:
        return {"error": f"No weather data for {city}. Known cities: {', '.join(WEATHER)}."}
    return {"city": city, **data}


def search_flights(origin: str, destination: str) -> dict:
    """Find flights between two cities, cheapest first.

    Args:
        origin: Departure city, for example "Mumbai".
        destination: Arrival city, for example "Goa".
    """
    matches = [f for f in FLIGHTS
               if f["origin"].lower() == origin.strip().lower()
               and f["destination"].lower() == destination.strip().lower()]
    return {"flights": sorted(matches, key=lambda f: f["price_inr"])}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between INR, USD, EUR and GBP.

    Args:
        amount: The amount to convert.
        from_currency: Currency code to convert from, for example "USD".
        to_currency: Currency code to convert to, for example "INR".
    """
    src, dst = from_currency.upper(), to_currency.upper()
    if src not in RATES_TO_INR or dst not in RATES_TO_INR:
        return {"error": f"Supported currencies: {', '.join(RATES_TO_INR)}."}
    converted = amount * RATES_TO_INR[src] / RATES_TO_INR[dst]
    return {"amount": amount, "from": src, "to": dst, "result": round(converted, 2)}


def book_flight(flight_id: str, passenger: str) -> dict:
    """Book a seat on a flight. This action spends money and cannot be undone.

    Args:
        flight_id: The flight id returned by search_flights, for example "FL-202".
        passenger: Full name of the passenger.
    """
    flight = next((f for f in FLIGHTS if f["id"] == flight_id), None)
    if flight is None:
        return {"error": f"Unknown flight id {flight_id}."}
    return {"confirmation": "BK-" + uuid.uuid4().hex[:6].upper(), "passenger": passenger, **flight}


# The same tools described for the model, in the JSON Schema format that the
# Chat Completions API expects. The model only ever sees these descriptions.
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string", "description": "City name, e.g. Goa"}},
                       "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_flights",
        "description": "Find flights between two cities, cheapest first. Prices are in INR.",
        "parameters": {"type": "object",
                       "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}},
                       "required": ["origin", "destination"]}}},
    {"type": "function", "function": {
        "name": "convert_currency",
        "description": "Convert an amount between INR, USD, EUR and GBP.",
        "parameters": {"type": "object",
                       "properties": {"amount": {"type": "number"},
                                      "from_currency": {"type": "string"},
                                      "to_currency": {"type": "string"}},
                       "required": ["amount", "from_currency", "to_currency"]}}},
]

# Name -> Python function: how our code finds the function the model asked for.
TOOLS = {"get_weather": get_weather, "search_flights": search_flights,
         "convert_currency": convert_currency}
