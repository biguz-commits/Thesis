FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

RUN pip install poetry
RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

RUN poetry install  --no-interaction --no-ansi --no-root


COPY ./app ./app

RUN poetry install --no-interaction --no-ansi


EXPOSE 8080

CMD exec uvicorn main:app --host 0.0.0.0 --port 8080