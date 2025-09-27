"""
All registered tools for LM Studio.
Just import this file and all tools are ready to use.
"""

from .registry import tool


@tool("Count the number of characters in a text string")
def char_counter(text: str) -> dict:
    """Count characters in the provided text."""
    return {
        "text": text,
        "character_count": len(text),
        "word_count": len(text.split()),
        "has_numbers": any(c.isdigit() for c in text)
    }


@tool("Perform mathematical calculations with two numbers")
def math_calculator(arg1: float, arg2: float, operation: str) -> dict:
    """
    Calculate result of mathematical operation.
    Operations: add, subtract, multiply, divide
    """
    operations = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else None
    }

    if operation not in operations:
        return {"error": f"Unknown operation: {operation}"}

    if operation == "divide" and arg2 == 0:
        return {"error": "Division by zero"}

    result = operations[operation](arg1, arg2)

    return {
        "arg1": arg1,
        "arg2": arg2,
        "operation": operation,
        "result": result
    }


@tool("Get the current weather for a city (mock data)")
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get mock weather data for demonstration."""
    # Mock data - in real app, would call weather API
    mock_weather = {
        "new york": {"temp": 22, "condition": "sunny"},
        "london": {"temp": 15, "condition": "rainy"},
        "tokyo": {"temp": 28, "condition": "cloudy"}
    }

    city_lower = city.lower()
    if city_lower in mock_weather:
        weather = mock_weather[city_lower]
        temp = weather["temp"]

        if units == "fahrenheit":
            temp = (temp * 9/5) + 32

        return {
            "city": city,
            "temperature": temp,
            "units": units,
            "condition": weather["condition"]
        }

    return {
        "city": city,
        "error": "City not found in mock data"
    }


@tool("Convert text to uppercase or lowercase")
def text_transformer(text: str, transform: str = "upper") -> dict:
    """Transform text case."""
    if transform == "upper":
        result = text.upper()
    elif transform == "lower":
        result = text.lower()
    elif transform == "title":
        result = text.title()
    else:
        return {"error": f"Unknown transform: {transform}"}

    return {
        "original": text,
        "transformed": result,
        "transform_type": transform
    }