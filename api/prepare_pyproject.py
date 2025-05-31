import os

TEMPLATE = "pyproject.template.toml"
OUTPUT = "pyproject.toml"

# Determine context: Docker build or local dev
# Default to local dev path
jules_core_path = os.environ.get("JULES_CORE_PATH", "../backend")

# If running in Docker, override to /app/backend
if os.environ.get("IN_DOCKER") == "1":
    jules_core_path = "/app/backend"

with open(TEMPLATE, "r") as f:
    content = f.read()

content = content.replace("{JULES_CORE_PATH}", jules_core_path)

with open(OUTPUT, "w") as f:
    f.write(content)

print(f"pyproject.toml generated with jules_core_path={jules_core_path}")
