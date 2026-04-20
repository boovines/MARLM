from pathlib import Path

from dotenv import load_dotenv

for candidate in (
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).resolve().parents[2] / ".env",
    Path(__file__).resolve().parents[3] / ".env",
    Path(__file__).resolve().parents[4] / ".env",
):
    if candidate.is_file():
        load_dotenv(candidate, override=False)
        break
