"""Check the Gemini connection without saving your API key."""
def main():
    # Importing this file during unittest discovery must never prompt or call Google.
    from getpass import getpass
    from google import genai

    key = getpass("Paste your Gemini API key (typing is hidden): ").strip()
    if not key:
        raise SystemExit("No key entered. Please run the test again.")

    try:
        with genai.Client(api_key=key) as client:
            first = next(iter(client.models.list()), None)
        if first is None:
            print("Connected, but no models were returned for this key.")
        else:
            print("Success! STARS connected to Gemini.")
    except Exception as exc:
        print(f"Connection failed ({type(exc).__name__}). Check your key and internet connection, then try again.")


if __name__ == '__main__':
    main()
