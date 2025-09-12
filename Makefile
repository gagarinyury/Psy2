up:        ## docker compose up
	docker compose up

down:      ## docker compose down -v
	docker compose down -v

migrate:   ## alembic upgrade head
	alembic upgrade head

test:      ## pytest -q
	pytest -q

run:       ## uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload