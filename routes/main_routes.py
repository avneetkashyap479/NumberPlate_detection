from flask import Blueprint, current_app, render_template, request

from database import db

main = Blueprint("main", __name__)


@main.get("/")
@main.get("/dashboard")
def dashboard():
    stats = db.get_stats(current_app.config["DATABASE_PATH"])
    jobs, _ = db.list_jobs(current_app.config["DATABASE_PATH"], page=1, per_page=6)
    analytics = db.get_analytics(current_app.config["DATABASE_PATH"])
    return render_template("dashboard.html", stats=stats, recent=jobs, analytics=analytics)


@main.get("/image")
def image_page():
    return render_template("image_detection.html")


@main.get("/video")
def video_page():
    return render_template("video_detection.html")


@main.get("/history")
def history_page():
    filters = {key: request.args.get(key, "") for key in ("search", "type", "date")}
    page = max(1, request.args.get("page", 1, type=int))
    jobs, total = db.list_jobs(current_app.config["DATABASE_PATH"], filters["search"], filters["type"], filters["date"], page)
    return render_template("history.html", jobs=jobs, total=total, page=page, filters=filters, per_page=20)


@main.get("/analytics")
def analytics_page():
    return render_template("analytics.html", stats=db.get_stats(current_app.config["DATABASE_PATH"]), analytics=db.get_analytics(current_app.config["DATABASE_PATH"]))


@main.get("/about")
def about_page():
    return render_template("about.html", model_path=str(current_app.config["MODEL_PATH"]))


@main.get("/job/<int:job_id>")
def job_page(job_id):
    job = db.get_job(current_app.config["DATABASE_PATH"], job_id)
    if not job:
        return render_template("404.html"), 404
    return render_template("job.html", job=job)
