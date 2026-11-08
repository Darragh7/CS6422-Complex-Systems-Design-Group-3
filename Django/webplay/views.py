import json
import logging
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
# Create your views here.

from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect, render
from django.views.decorators.http import require_POST


logger = logging.getLogger(__name__)


class AuthenticationBackendUnavailable(Exception):
    pass


def _decode_backend_response(body):
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _call_auth_backend(endpoint, payload):
    request = Request(
        f"{settings.BACKEND_API_URL}/users/{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=5) as response:
            return response.status, _decode_backend_response(response.read())
    except HTTPError as error:
        return error.code, _decode_backend_response(error.read())
    except (URLError, TimeoutError, OSError) as error:
        logger.error("Authentication backend request failed")
        raise AuthenticationBackendUnavailable from error


def _store_account_session(request, response_data):
    if not isinstance(response_data, dict):
        raise AuthenticationBackendUnavailable("Invalid authentication response")
    account = response_data.get("user")
    if (
        not isinstance(account, dict)
        or not isinstance(account.get("user_id"), int)
        or not isinstance(account.get("first_name"), str)
    ):
        raise AuthenticationBackendUnavailable("Invalid authentication response")

    request.session.cycle_key()
    request.session["user_id"] = account["user_id"]
    request.session["first_name"] = account["first_name"]


def landing(request):
    return render(request, "land2.html")


@require_POST
def signup(request):
    payload = {
        "first_name": request.POST.get("first_name", ""),
        "last_name": request.POST.get("last_name", ""),
        "email": request.POST.get("email", ""),
        "phone": request.POST.get("phone", ""),
        "password": request.POST.get("password", ""),
    }
    try:
        status_code, response_data = _call_auth_backend("signup", payload)
        if status_code == 201:
            _store_account_session(request, response_data)
            messages.success(request, "Your account has been created.")
        elif status_code == 409:
            messages.error(request, "An account with that email or phone already exists.")
        elif status_code == 422:
            messages.error(
                request,
                "Check your first and last name, enter a valid email or supported "
                "phone number, and use a password of 8–128 characters. Phone "
                "formats: +353 8XXXXXXXX or +91 XXXXXXXXXX.",
            )
        else:
            messages.error(request, "We could not create your account. Please try again.")
    except AuthenticationBackendUnavailable:
        messages.error(request, "Sign up is temporarily unavailable. Please try again.")

    return redirect("landing")


@require_POST
def login(request):
    payload = {
        "identifier": request.POST.get("identifier", ""),
        "password": request.POST.get("password", ""),
    }
    try:
        status_code, response_data = _call_auth_backend("login", payload)
        if status_code == 200:
            _store_account_session(request, response_data)
            messages.success(request, "You are signed in.")
        elif status_code == 401:
            messages.error(request, "Invalid email, phone, or password.")
        elif status_code == 422:
            messages.error(request, "Enter your email or phone and password.")
        else:
            messages.error(request, "We could not sign you in. Please try again.")
    except AuthenticationBackendUnavailable:
        messages.error(request, "Sign in is temporarily unavailable. Please try again.")

    return redirect("landing")


@require_POST
def logout(request):
    request.session.flush()
    messages.success(request, "You are signed out.")
    return redirect("landing")


def home(request):
    return render(request, "landing.html")
