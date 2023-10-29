"""Base state for the app."""

import reflex as rx

class User(rx.Model, table=True):
    name: str
    age: int 
    email: str
    phone: str
    pfp: str
    tumor: str


class State(rx.State):
    """Base state for the app.

    The base state is used to store general vars used throughout the app.
    """
    registered: bool = False
    name: str = ""
    pfp: str
    tumor: str

    def handle_submit(self, form_data: dict):
        print("handling")
        self.registered = True
        self.name = form_data.get("name")
        with rx.session() as session:
            session.add(
                User(
                    name=form_data.get("name"),
                    age=form_data.get("age"),
                    email=form_data.get("email"),
                    phone=form_data.get("phone"),
                    pfp="pfp",
                    tumor="tumor",
                    )
                )
            session.commit()
        rx.redirect("/matches")