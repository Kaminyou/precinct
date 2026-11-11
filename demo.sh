uv run python -m precinct --M 300 --duration 10
uv run python -m precinct --M 300 --centralized --duration 30
uv run python -m precinct.viz.server
# uv run python -m precinct.viz.server --scenario demo if time permits