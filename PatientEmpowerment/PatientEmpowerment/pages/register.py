"""The dashboard page."""
from PatientEmpowerment.templates import template
from PatientEmpowerment.state import State

import reflex as rx


@template(route="/register", title="Register")
def register() -> rx.Component:
    return rx.vstack(
        rx.form(
            rx.vstack(
                rx.input(
                    placeholder="First Name",
                    id="first_name",
                ),
                rx.input(
                    placeholder="Last Name", id="last_name"
                ),
                rx.input(
                    placeholder="Phone #", id="phone"
                ),
                rx.input(
                    placeholder="Email Address", id="email"
                ),
                rx.upload(
                    rx.text(
                        "Drag and drop files here or click to select files"
                    ),
                    border="1px dotted rgb(107,99,246)",
                    padding="5em",
                ),
                rx.button("Submit", type_="submit"),
            ),
            on_submit = State.handle_submit,
        ),
    )
