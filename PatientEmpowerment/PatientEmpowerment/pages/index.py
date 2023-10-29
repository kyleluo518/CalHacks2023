"""The home page of the app."""

from PatientEmpowerment import styles
from PatientEmpowerment.templates import template

import reflex as rx


@template(route="/", title="Home", image="/github.svg")
def index() -> rx.Component:
    """The home page.

    Returns:
        The UI for the home page.
    """
    return rx.box(
        rx.heading("Hi, we are Patient Empowerment!"),
        rx.text("Metrics:"),
        rx.image(src="/segmodel_explanation.png", height="30em")
    )
