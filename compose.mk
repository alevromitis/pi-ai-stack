# Included by every service's Makefile. Compose doesn't discover
# docker-compose.<service>.yml, so the -f lives here, once (issue #69).

# Repo root, from this file's own path (read before any other include).
REPO_ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

COMPOSE_FILE := $(wildcard docker-compose.*.yml)
ifneq ($(words $(COMPOSE_FILE)),1)
$(error $(CURDIR): expected exactly one docker-compose.*.yml, found '$(COMPOSE_FILE)')
endif
COMPOSE := docker compose -f $(COMPOSE_FILE)
SECRETS := $(REPO_ROOT)/scripts/secrets.sh

.PHONY: up down pull config ps logs restart encrypt decrypt

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

encrypt: ## SOPS-encrypt this stack's .env / .env.* into *.sops (public key only)
	$(SECRETS) encrypt $(CURDIR)

decrypt: ## Write .env / .env.* (mode 600) from *.sops; needs the age private key
	$(SECRETS) decrypt $(CURDIR)
