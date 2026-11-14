import json
import urllib.request

# Four recommended sights for each city that the catalog knows.
ATTRACTIONS = {
    "kyoto": ["Fushimi Inari Taisha", "Kinkaku-ji", "Arashiyama Bamboo Grove", "Gion district"],
    "jaipur": ["Amber Fort", "City Palace", "Hawa Mahal", "Johari Bazaar"],
    "goa": ["Basilica of Bom Jesus", "Fontainhas quarter", "Palolem Beach", "Dudhsagar Falls"],
}

# The typical cost of one day for one person in US dollars: a share of a
# mid-range room, meals, local transport and entry fees. Illustrative values.
DAILY_COST_PER_PERSON_USD = {"kyoto": 160, "jaipur": 60, "goa": 75}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount using a live FX rate from a public API.

    Args:
        amount: The amount to convert.
        from_currency: ISO code such as "USD".
        to_currency: ISO code such as "INR".

    Returns:
        A dictionary whose "status" is "success" with the converted amount and
        exchange rate, or "error" with an "error_message".
    """
    try:
        amount_value = float(amount)
    except (TypeError, ValueError):
        return {"status": "error", "error_message": "amount must be a number."}

    if amount_value < 0:
        return {"status": "error", "error_message": "amount must be zero or greater."}

    from_code = from_currency.strip().upper()
    to_code = to_currency.strip().upper()
    if len(from_code) != 3 or len(to_code) != 3:
        return {
            "status": "error",
            "error_message": "Currency codes must be 3-letter ISO codes like USD or INR.",
        }

    try:
        url = f"https://api.frankfurter.app/latest?from={from_code}&to={to_code}"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.load(response)
    except Exception:
        return {
            "status": "error",
            "error_message": f"Could not fetch a live exchange rate for {from_code} to {to_code}.",
        }

    rates = data.get("rates", {})
    if to_code not in rates:
        return {
            "status": "error",
            "error_message": f"No live rate is available for {from_code} to {to_code}.",
        }

    rate = float(rates[to_code])
    converted = amount_value * rate
    return {
        "status": "success",
        "amount": amount_value,
        "from_currency": from_code,
        "to_currency": to_code,
        "rate": rate,
        "converted_amount": converted,
    }


def get_top_attractions(city: str) -> dict:
    """Return the recommended sights of a city.

    Args:
        city: The name of the city, for example "Kyoto".

    Returns:
        A dictionary whose "status" is "success" together with a list of
        sights, or "error" together with an "error_message".
    """
    # The catalog keys are lowercase, so the city name is normalized first.
    attractions = ATTRACTIONS.get(city.strip().lower())
    if attractions is None:
        # An error is returned as data, not raised as an exception. The model
        # reads the message and can explain the problem to the user.
        return {
            "status": "error",
            "error_message": f"The catalog has no entry for {city}. Known cities: Kyoto, Jaipur, Goa.",
        }
    return {"status": "success", "city": city, "attractions": attractions}


def estimate_trip_cost(city: str, nights: int, travelers: int) -> dict:
    """Estimate the total cost of a trip in US dollars, excluding flights.

    Args:
        city: The destination city, for example "Kyoto".
        nights: The number of nights, at least 1.
        travelers: The number of travelers, at least 1.

    Returns:
        A dictionary whose "status" is "success" together with the total
        cost and the values used to compute it, or "error" together with an
        "error_message".
    """
    daily_cost = DAILY_COST_PER_PERSON_USD.get(city.strip().lower())
    if daily_cost is None:
        return {
            "status": "error",
            "error_message": f"No cost data for {city}. Known cities: Kyoto, Jaipur, Goa.",
        }
    if nights < 1 or travelers < 1:
        return {"status": "error", "error_message": "nights and travelers must both be at least 1."}

    # A trip of n nights has n + 1 days. The arithmetic happens here, in code,
    # because a language model can make mistakes when it calculates.
    days = nights + 1
    total = daily_cost * days * travelers
    return {
        "status": "success",
        "city": city,
        "days": days,
        "travelers": travelers,
        "cost_per_person_per_day_usd": daily_cost,
        "total_usd": total,
    }
