Can you add unison to the ow-* images?

What I was doing

- Running the ariahw/rl-rewardhacking verl stack on a raw pod via ow ssh --sync --existing
- Image was nielsrolf/ow-vllm:v0.11

What happened

[ow] unison not found on remote. Please install it in your image.
Connection to <pod> closed.

Why I think this is an image thing, not a me thing

- openweights/cli/common.py:89 already documents unison as a requirement on both ends
- common.py:217 is the check that fires, so the CLI knows it needs it
- The ow-* images just don't ship it, so --sync can't work out of the box on any of them

The ask

- Add unison to the ow-* images (one apt line?)
- And say somewhere which version the image ends up with

The unison version also matters: unison refuses to talk to a different version on the other end, so people need to know what to install locally to match. Ubuntu apt and macOS brew also may not agree.

What I can do is clone the repo directly on the pod and scp my .env over. Fine for now, but I lose live sync, which I would like back once I am iterating on training code.
