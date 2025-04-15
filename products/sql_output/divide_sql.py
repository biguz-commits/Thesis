import os

# Percorsi relativi al file Python
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR)
OUTPUT_FOLDER = os.path.join(BASE_DIR, "chunks")
LINES_PER_CHUNK = 5000  # Personalizzabile

# Crea cartella per i file spezzati
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Prende tutti i file .sql nella cartella
sql_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".sql")]

for sql_file in sql_files:
    input_path = os.path.join(INPUT_FOLDER, sql_file)

    with open(input_path, "r") as infile:
        lines = infile.readlines()

    for i in range(0, len(lines), LINES_PER_CHUNK):
        chunk_lines = lines[i:i + LINES_PER_CHUNK]
        chunk_filename = f"{os.path.splitext(sql_file)[0]}_part_{i // LINES_PER_CHUNK + 1}.sql"
        chunk_path = os.path.join(OUTPUT_FOLDER, chunk_filename)

        with open(chunk_path, "w") as chunk_file:
            chunk_file.writelines(chunk_lines)

print("Divisione completata. File salvati in:", OUTPUT_FOLDER)
