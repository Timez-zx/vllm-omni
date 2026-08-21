# live_agent benchmark

The supported benchmark is the playback-paced continuous AV workload in
`web_client/`. The stateful WebSocket submits silent finite Thinker warm-ups as
accepted frames arrive, then one finite response request per turn. All requests
may reuse disposable prefix KV opportunistically.

```text
web_client/  canonical client, browser, workload preparation, and capacity runner
harness/     shared GPU sampler plus older diagnostic drivers
analysis/    result verification and root-cause reports
```

Start with `web_client/README.md`. Historical harnesses and reports are not part
of the current capacity curve unless that document names them explicitly.
