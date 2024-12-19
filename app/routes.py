from flask import render_template, request
from src.api.LLM_Config import generate_sql
from src.database.DB_Config import connect_to_db
from app import create_app

app = create_app()

@app.route("/", methods=["GET", "POST"])
def home():
    sql_query = ""
    result = []
    if request.method == "POST":
        user_query = request.form["user_query"]
        sql_query = generate_sql(user_query)  # This uses the Gemini 1.5 model to generate SQL
        
        db_path = "data/sampledata.db"
        conn = connect_to_db(db_path)
        try:
            result = conn.execute(sql_query).fetchall()
        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", sql_query=sql_query, result=result)

if __name__ == "__main__":
    app.run(debug=True)
