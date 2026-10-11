include $(shell git rev-parse --show-toplevel)/compose.mk

# The sandboxes' network is external, so `make down` leaves it: a kept
# (stopped) sandbox starts only on the network it was created on, since
# Docker holds the network's id, not its name.
up: egress-network

.PHONY: egress-network
egress-network: ## Create agent-sandbox-egress if it doesn't exist
	@docker network inspect agent-sandbox-egress >/dev/null 2>&1 \
		|| docker network create --driver bridge --subnet 10.89.32.0/24 agent-sandbox-egress
