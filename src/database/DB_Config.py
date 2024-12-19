import sqlite3

def connect_to_db(db_path):
    connection = sqlite3.connect(db_path)
    return connection
