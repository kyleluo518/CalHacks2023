"""Base state for the app."""

import reflex as rx

class State(rx.State):
    """Base state for the app.

    The base state is used to store general vars used throughout the app.
    """
    form_data = {}

    def handle_submit(self, form_data: dict):
        self.form_data = form_data
