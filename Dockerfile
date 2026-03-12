FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN pip install uv && uv sync --package kinship-webapp

EXPOSE 8000

CMD ["uv", "run", "--package", "kinship-webapp", "python", "-m", "webapp.app"]
