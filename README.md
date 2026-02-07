# VoxtralFoundation

Swift Package that vendors the C implementation from `antirez/voxtral.c` and exposes it as a C module target.

## Original Source
- Upstream repository: `https://github.com/antirez/voxtral.c`
- The full upstream source is vendored in `vendor/voxtral.c`.
- The package compiles a curated copy into `Sources/VoxtralC` to keep SwiftPM build inputs explicit and stable.

## Rationale
- Provide a single SwiftPM dependency that builds the C core with minimal friction.
- Keep the upstream code intact and easy to update while allowing Swift-friendly consumption.
- Enable Metal acceleration on Apple platforms without requiring consumers to manage Objective‑C / Metal build flags.

## Notes
- Metal acceleration is enabled on Apple platforms via `voxtral_metal.m`.
- On non-Apple platforms, Metal APIs are stubbed out (`voxtral_metal_stub.c`).
- BLAS uses Accelerate on Apple platforms; a naive fallback is used elsewhere.

## Usage
Add this package and import `VoxtralC` from Swift or C targets.
