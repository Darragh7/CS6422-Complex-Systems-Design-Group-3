from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse


class AuthenticationViewTests(TestCase):

    @patch("webplay.views._call_auth_backend")
    def test_signup_creates_session_after_backend_success(self, call_backend):
        call_backend.return_value = (
            201,
            {"user": {"user_id": 123, "first_name": "Ada"}},
        )

        response = self.client.post(
            reverse("signup"),
            {
                "first_name": "Ada",
                "last_name": "Lovelace",
                "email": "ada@example.com",
                "password": "correct horse battery staple",
            },
        )

        self.assertRedirects(response, reverse("landing"))
        self.assertEqual(self.client.session["user_id"], 123)
        call_backend.assert_called_once_with(
            "signup",
            {
                "first_name": "Ada",
                "last_name": "Lovelace",
                "email": "ada@example.com",
                "phone": "",
                "password": "correct horse battery staple",
            },
        )

    @patch("webplay.views._call_auth_backend")
    def test_login_rejects_invalid_credentials_without_creating_session(
        self, call_backend
    ):
        call_backend.return_value = (
            401,
            {"detail": "Invalid email, phone, or password"},
        )

        response = self.client.post(
            reverse("login"),
            {"identifier": "ada@example.com", "password": "wrong password"},
        )

        self.assertRedirects(response, reverse("landing"))
        self.assertNotIn("user_id", self.client.session)

    @patch("webplay.views._call_auth_backend")
    def test_login_creates_session_after_backend_success(self, call_backend):
        call_backend.return_value = (
            200,
            {"user": {"user_id": 321, "first_name": "Ada"}},
        )

        response = self.client.post(
            reverse("login"),
            {"identifier": "ada@example.com", "password": "correct password"},
        )

        self.assertRedirects(response, reverse("landing"))
        self.assertEqual(self.client.session["user_id"], 321)

    def test_logout_clears_the_account_session(self):
        session = self.client.session
        session["user_id"] = 321
        session["first_name"] = "Ada"
        session.save()

        response = self.client.post(reverse("logout"))

        self.assertRedirects(response, reverse("landing"))
        self.assertNotIn("user_id", self.client.session)

    def test_landing_page_forms_include_csrf_tokens(self):
        response = self.client.get(reverse("landing"))

        self.assertContains(response, 'action="/webplay/login/"')
        self.assertContains(response, 'action="/webplay/signup/"')
        self.assertContains(response, 'name="csrfmiddlewaretoken"')
