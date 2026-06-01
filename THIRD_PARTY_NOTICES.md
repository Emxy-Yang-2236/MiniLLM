# Third-Party Notices

## Stanford CS336 Assignment 1

Repository name: `stanford-cs336/assignment1-basics`

Source URL: `https://github.com/stanford-cs336/assignment1-basics`

License: MIT

MiniLLM vendors or adapts portions of Stanford CS336 Assignment 1 for public-test compatibility. The vendored material is used as a strict compatibility target for tokenizer/model/training/checkpoint correctness. MiniLLM-specific SFT, pipeline, systems benchmark and report requirements are separate project extensions.

Copied/adapted components:

- public tests
- fixtures
- snapshots, where present
- adapter pattern and local bridge design

Local locations:

- `shared/tests/cs336_a1_exact/`
- `shared/tests/cs336_a1_exact/fixtures/`
- `shared/tests/cs336_a1_exact/_snapshots/`
- `shared/tests/cs336_a1_exact/adapters.py`
- `shared/tests/fixtures/`

This project is not affiliated with, sponsored by, or endorsed by Stanford University unless a separate course announcement explicitly states otherwise.

## MIT License Notice

```text
Copyright 2025 Stanford University

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

Some assignment text was converted from Stanford CS336 Assignment 1 handout for course-development/reference purposes. Converted local materials include:

- `cs336_assignment1_basics.pdf`
- `docs/cs336_a1_section2.md`
- `docs/cs336_a1_section3.md`
- `docs/cs336_a1_section2_3_conversion_notes.md`

See original CS336 materials for authoritative wording.

## TinyStories

Dataset name: TinyStories V2 GPT-4

Source URL: `https://huggingface.co/datasets/roneneldan/TinyStories`

License noted by this project: CDLA-Sharing-1.0

Local generated/release metadata:

- `data/full_release/README.md`
- `data/full_release/manifest.json`

TinyStories data is not relicensed by the MiniLLM MIT License. Users should follow the dataset license and attribution requirements from the upstream dataset source.
