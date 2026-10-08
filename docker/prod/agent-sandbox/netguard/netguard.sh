#!/bin/sh
#
# Host-side network limits for the agent sandboxes, on their bridge
# (SANDBOX_BRIDGE), from the host's network namespace - outside every
# sandbox, so root inside one can't lift them:
#
#   nftables  table inet agent_sandbox: public IPv4 internet only (no LAN,
#             tailnet, host, other containers or IPv6), no SMTP, nothing
#             connecting in; per sandbox, caps on new and open connections.
#   tc        all sandboxes together: SANDBOX_NET_DOWN towards them (htb +
#             fq_codel on the bridge), SANDBOX_NET_UP from them (policer on
#             the bridge's ingress).
#
#   netguard run     apply both, then re-apply whatever goes missing (a
#                    reboot clears tc, `nft flush ruleset` clears nft)
#   netguard check   healthcheck; the gateway starts no sandbox without it
#   netguard remove  delete both (by hand, after `make down`)
#
# Rules stay when the container stops: failing closed is the point.

set -eu

BR=${SANDBOX_BRIDGE:-br-agentsbx}
DOWN=${SANDBOX_NET_DOWN:-20mbit}
UP=${SANDBOX_NET_UP:-10mbit}
RATE=${SANDBOX_NEW_CONN_RATE:-20}
MAXCONN=${SANDBOX_MAX_CONN:-256}

ruleset() {
  # The first two lines make the load atomic and idempotent: the table is
  # replaced as a whole, or nothing changes.
  cat <<EOF
table inet agent_sandbox
delete table inet agent_sandbox
table inet agent_sandbox {
	# Everything that is not the public internet: RFC 1918, CGNAT (the
	# tailnet), loopback, link-local, documentation, benchmarking,
	# multicast and reserved.
	set not_internet {
		type ipv4_addr
		flags interval
		elements = {
			0.0.0.0/8, 10.0.0.0/8, 100.64.0.0/10, 127.0.0.0/8,
			169.254.0.0/16, 172.16.0.0/12, 192.0.0.0/24, 192.0.2.0/24,
			192.168.0.0/16, 198.18.0.0/15, 198.51.100.0/24,
			203.0.113.0/24, 224.0.0.0/3
		}
	}

	# Keyed by sandbox address.
	set new_conns {
		type ipv4_addr
		size 1024
		flags dynamic,timeout
		timeout 1m
	}
	set open_conns {
		type ipv4_addr
		size 1024
		flags dynamic
	}

	# Ahead of Docker's own chains (priority filter = 0). A drop here is
	# final; an accept still passes through Docker's.
	chain forward {
		type filter hook forward priority filter - 10; policy accept;
		# Nothing opens a connection to a sandbox, from anywhere - another
		# sandbox included (enable_icc=false makes that traffic visible here).
		oifname "$BR" ct state new counter drop
		iifname "$BR" jump from_sandbox
	}

	chain from_sandbox {
		ct state established,related return
		meta nfproto ipv6 counter drop
		# Post-DNAT address, so a published port on the host's LAN address
		# is caught as the container address it is forwarded to.
		ip daddr @not_internet counter drop
		# No mail server of its own: port 25 is where abuse reports come from.
		tcp dport 25 counter drop
		ct state new add @new_conns { ip saddr limit rate over $RATE/second burst $((RATE * 2)) packets } counter drop
		ct state new add @open_conns { ip saddr ct count over $MAXCONN } counter drop
	}

	# The host itself (sshd, Tailscale, anything listening on it): nothing
	# from the bridge is for it. Docker's DNS server lives in each
	# container's own namespace, so this doesn't reach it.
	chain input {
		type filter hook input priority filter - 10; policy accept;
		iifname "$BR" counter drop
	}
}
EOF
}

apply_nft() {
  ruleset | nft -f -
  echo "netguard: nftables table inet agent_sandbox loaded"
}

apply_tc() {
  # Towards the sandboxes: the host routes their downloads into the bridge,
  # through its root qdisc.
  tc qdisc replace dev "$BR" root handle 1: htb default 10
  tc class replace dev "$BR" parent 1: classid 1:10 htb rate "$DOWN" ceil "$DOWN" quantum 1514
  tc qdisc replace dev "$BR" parent 1:10 handle 10: fq_codel
  # From them: what the bridge hands up to the host to route out.
  tc qdisc del dev "$BR" ingress 2>/dev/null || true
  tc qdisc add dev "$BR" handle ffff: ingress
  tc filter add dev "$BR" parent ffff: protocol ip prio 1 u32 match u32 0 0 \
    police rate "$UP" burst 64k drop flowid :1
  echo "netguard: $BR shaped to $DOWN down, $UP up"
}

nft_ok() {
  nft list table inet agent_sandbox >/dev/null 2>&1
}

tc_ok() {
  qdiscs=$(tc qdisc show dev "$BR" 2>/dev/null) || return 1
  case $qdiscs in *"qdisc htb 1: root"*) ;; *) return 1 ;; esac
  case $qdiscs in *"qdisc ingress ffff:"*) ;; *) return 1 ;; esac
}

bridge_up() {
  ip link show dev "$BR" >/dev/null 2>&1
}

run() {
  trap 'exit 0' TERM INT
  # Always load once at start, so changed limits in .env take effect on
  # `make up`; after that, only what has gone missing.
  apply_nft || echo "netguard: loading nftables failed" >&2
  loaded_tc=false
  while :; do
    nft_ok || apply_nft || echo "netguard: loading nftables failed" >&2
    if bridge_up; then
      if ! $loaded_tc || ! tc_ok; then
        if apply_tc; then loaded_tc=true; else echo "netguard: tc on $BR failed" >&2; fi
      fi
    else
      echo "netguard: waiting for bridge $BR (the egress network)" >&2
    fi
    sleep 15 &
    wait $!
  done
}

case ${1:-run} in
  run) run ;;
  check) nft_ok && bridge_up && tc_ok ;;
  remove)
    nft delete table inet agent_sandbox 2>/dev/null || true
    tc qdisc del dev "$BR" root 2>/dev/null || true
    tc qdisc del dev "$BR" ingress 2>/dev/null || true
    echo "netguard: removed" ;;
  *) echo "usage: netguard [run|check|remove]" >&2; exit 2 ;;
esac
