"""The settings page."""

from PatientEmpowerment.templates import template

import reflex as rx


@template(route="/matches", title="Matches")
def matches() -> rx.Component:
    """The settings page.

    Returns:
        The UI for the settings page.
    """
    return rx.vstack(
        rx.heading("Settings", font_size="3em"),
        rx.text("Welcome to Reflex!"),
        rx.text(
            "You can edit this page in ",
            rx.code("{your_app}/pages/settings.py"),
        ),
    )