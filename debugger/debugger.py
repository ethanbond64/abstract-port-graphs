from flask import Flask, request, render_template, send_file
import os

from arc.arc_objects import ArcInterpreter
from arc.arc_utils import ArcTask
from debugger.debugger_state import DebuggerState
from arc.arc_solutions import *


def create_app():

    app = Flask(__name__, template_folder="debugger_templates")

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":

            challenge_id = request.form.get("challenge_id")

            if challenge_id is None:
                return "Challenge ID is required", 400

            function_name = "solve_" + challenge_id
            graph_function = globals().get(function_name)
            new_graph = graph_function()
            challenge = ArcTask.get(challenge_id)

            interpreter = ArcInterpreter()
            DebuggerState.initialize(challenge, 0, new_graph, interpreter.solve)

        debug_info = DebuggerState.get_info()
        return render_template("index.html",
                               challenge_id=debug_info.get("challenge_id"),
                               data=str(debug_info))

    @app.route("/step", methods=["GET", "POST"])
    def step():
        DebuggerState.set_debugger_commands(0, 0)
        debug_info = DebuggerState.get_info()
        return render_template("index.html",
                               challenge_id=debug_info.get("challenge_id"),
                               data=str(debug_info))

    @app.route("/graph", methods=["GET"])
    def graph():
        uri = DebuggerState.get_graph_url()
        if os.path.exists(uri):
            return send_file(uri, mimetype="text/html")
        else:
            return "File not found", 404

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.run()
