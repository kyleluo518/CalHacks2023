"""The settings page."""

from PatientEmpowerment.templates import template
from PatientEmpowerment.state import State, User

import reflex as rx


@template(route="/matches", title="Matches")
def matches() -> rx.Component:
    """The settings page.

    Returns:
        The UI for the settings page.
    """
    users = [
        {
            "name": "Jason",
            "phone": "650-555-5555",
            "email": "example@example.com",
            "age": 40,
            "pic": rx.image(src="/person.jpeg", width="25em"),
        },
        {
            "name": "Mary",
            "phone": "650-555-5555",
            "email": "example@example.com",
            "age": 41,
            "pic": rx.image(src="/person.jpeg", width="25em"),
        },
    ]
    return rx.container(
        rx.cond(
            ~State.registered,
            rx.heading("Please register to view matches.")
        ),
        rx.cond(
            State.registered,
            rx.vstack(
            *[
                rx.card(
                    rx.box(
                        rx.text(person["age"]),
                        rx.text(person["email"]),
                        rx.text(person["phone"]),
                        person["pic"],
                    ),
                    header=rx.heading(person["name"]),
                )
                for person in users
            ],
            rx.circular_progress(is_indeterminate=True),
            spacing = "5px",
            )

        )
    )
