# Included by every service's Makefile. Compose doesn't discover
# docker-compose.<service>.yml, so the -f lives here, once (issue #69).
COMPOSE_FILE := $(wildcard docker-compose.*.yml)
ifneq ($(words $(COMPOSE_FILE)),1)
$(error $(CURDIR): expected exactly one docker-compose.*.yml, found '$(COMPOSE_FILE)')
endif
COMPOSE := docker compose -f $(COMPOSE_FILE)

.PHONY: up down pull config ps logs restart

up:      ## Start or update the stack, detached
	$(COMPOSE) up -d

down:    ## Stop and remove the stack's containers
	$(COMPOSE) down

pull:    ## Pull the stack's images
	$(COMPOSE) pull

config:  ## Render and validate the resolved Compose config
	$(COMPOSE) config

ps:      ## List the stack's containers
	$(COMPOSE) ps

logs:    ## Follow the stack's logs
	$(COMPOSE) logs -f --tail=100

restart: ## Restart the stack's containers
	$(COMPOSE) restart
