"""The dashboard page."""
from PatientEmpowerment.templates import template
from PatientEmpowerment.state import State

import reflex as rx


@template(route="/register", title="Register")
def register() -> rx.Component:
    return rx.vstack(
        rx.cond(
            State.registered,
            rx.heading("Thank you for registrating!")
        ),
        rx.cond(
            ~State.registered,
            rx.form(
                rx.vstack(
                    rx.input(
                        placeholder="Name", id="name"
                    ),
                    rx.input(
                        placeholder="Age", id="age"
                    ),
                    rx.input(
                        placeholder="Phone #", id="phone"
                    ),
                    rx.input(
                        placeholder="Email Address", id="email"
                    ),
                    rx.text("Picture of you"),
                    rx.upload(
                        rx.text(
                            "Drag and drop files here or click to select files"
                        ),
                        border="1px dotted rgb(107,99,246)",
                        padding="5em",
                        id="pfp"
                    ),
                    rx.text("Picture of MRI"),
                    rx.upload(
                        rx.text(
                            "Drag and drop files here or click to select files"
                        ),
                        border="1px dotted rgb(107,99,246)",
                        padding="5em",
                        id="tumor"
                    ),
                    rx.button("Submit", type_="submit"),
                ),
                on_submit = State.handle_submit,
            ),
        )
    )
