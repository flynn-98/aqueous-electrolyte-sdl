# dependencies
import scientia_sdk as sct


SDLABS_ENDPOINT_URL = "https://api.enterprise.atinary.com/sdlabs/latest"  # as found in the documentation (SDLabs SDK > API Endpoints)
SDLABS_API_KEY = "eyJhbGciOiJIUzUxMiIsImtpZCI6ImtleV8yZTNiZTUyYzZkZGQ0N2UzYjA1NGY4NTQ0N2JmZTBlMiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL2F1dGguYXRpbmFyeS5jb20iLCJjb2duaXRvOmdyb3VwcyI6WyJDQVBlWF9QaW9uZWVyX0NlbnRlciJdLCJpYXQiOjE3NTk4NDkzMDMsIm5iZiI6MTc1OTg0OTMwMywidXNlcm5hbWUiOiJmMmM2ZDBiYy01OTQ1LTRiM2UtYjA3Mi0yMzc5ZTI1YmI0NjgifQ.ia3l2NFLFEp-WmDyJnRkvpCwUJEeDt_3czWCA3cRCF-8qEOujy0kEJxGK5ow5kVr3mFzADjRkioCIk-_5TpJvA"
SDLABS_GROUP_NAME = "CAPeX_Pioneer_Center"  


def client():
    # Create SDLabs API client
    configuration = sct.Configuration(
        host=SDLABS_ENDPOINT_URL,
        api_key={"api_key": SDLABS_API_KEY},
    )
    configuration.access_token = None
    sdlabs_api_client = sct.ApiClient(configuration)

    return sdlabs_api_client

