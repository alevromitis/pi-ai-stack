# syntax=docker/dockerfile:1
#
# netguard: nftables and tc on the sandbox bridge, from the host's network
# namespace (see netguard.sh). Its own nft table, so the nft binary here
# needn't match the host's iptables or Docker's firewall backend.
#
# Built on rpi (arm64).
FROM debian:trixie-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 \
        nftables \
    && rm -rf /var/lib/apt/lists/*

COPY --chmod=755 netguard.sh /usr/local/bin/netguard

ENTRYPOINT ["netguard"]
CMD ["run"]
