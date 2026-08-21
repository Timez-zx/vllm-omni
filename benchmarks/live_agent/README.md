# live_agent benchmark

The supported benchmark is the playback-paced continuous AV workload in
`web_client/`. It measures a stateful WebSocket service whose model engine
receives one finite request per turn and may reuse prefix KV opportunistically.

```text
web_client/  canonical client, browser, workload preparation, and capacity runner
harness/     shared GPU sampler plus older diagnostic drivers
analysis/    result verification and root-cause reports
```

Start with `web_client/README.md`. Historical harnesses and reports are not part
of the current capacity curve unless that document names them explicitly.
