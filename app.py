import logging
from pathlib import Path

from flask import Flask, render_template

from config.settings import Settings, ensure_directories
from database.db import initialize_database
from routes.api_routes import api
from routes.main_routes import main


def create_app(test_config=None):
    app = Flask(__name__)
    app.config.from_object(Settings)
    app.config["SETTINGS"] = Settings
    if test_config:
        app.config.update(test_config)
        app.config["SETTINGS"] = test_config.get("SETTINGS", Settings)
    ensure_directories(app.config["SETTINGS"])
    initialize_database(app.config["DATABASE_PATH"])
    app.register_blueprint(main)
    app.register_blueprint(api)

    @app.get("/files/<int:job_id>/<kind>")
    def job_output(job_id, kind):
        from flask import redirect, url_for
        if kind == "annotated":
            return redirect(url_for("api.detection_file", job_id=job_id, kind=kind, detection_id=0))
        if kind == "video":
            return redirect(url_for("api.detection_file", job_id=job_id, kind=kind, detection_id=0))
        return render_template("404.html"), 404

    @app.errorhandler(404)
    def not_found(_error):
        return render_template("404.html"), 404

    @app.errorhandler(413)
    def too_large(_error):
        return render_template("404.html", message="Upload exceeds the configured size limit."), 413

    if not app.testing:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
