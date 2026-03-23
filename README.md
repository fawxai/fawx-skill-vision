# Vision Skill

WASM skill plugin for [Fawx](https://github.com/fawxai/fawx).

## Install

```bash
fawx skill install fawxai/vision
```

## Build from Source

```bash
cargo build --release --target wasm32-unknown-unknown
fawx skill install ./target/wasm32-unknown-unknown/release/vision_skill.wasm
```

## License

Apache 2.0
