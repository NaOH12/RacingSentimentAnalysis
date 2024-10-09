import sqlite3

file_name = "ACCReplay_240223-042732_H_Barcelona.rpy"
# Attempt to open file as an sql database file
try:
    conn = sqlite3.connect(file_name)
    print("File opened successfully")
    # Get data
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for table in tables:
        print(table)
    conn.close()
except sqlite3.Error as e:
    print("Error opening file: ", e)
    conn.close()