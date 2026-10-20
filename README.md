*This project has been created as part of the 42 curriculum by romarti2.*

# call me maybe

Function calling with a small language model (Qwen3-0.6B) using constrained decoding.

## Description

**Goal.** Given a natural-language prompt such as `"What is the sum of 40 and 2?"`, the program does not answer the question. It returns the function call that would solve it: the function name and its arguments with the right types.

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

**Overview.** A 0.6B-parameter model is unreliable at producing JSON when it is only prompted for it. This project does not rely on the prompt for structure. At every generation step the logits returned by the model are modified *before* a token is chosen, so that only tokens that keep the output valid JSON and compliant with the function schema can be selected. The model decides the content (which function, which values). The code guarantees the structure.

Inputs (in `data/input/` by default):

- `functions_definition.json`: the available functions, with parameter names and types.
- `function_calling_tests.json`: the prompts to process.

Output: `data/output/function_calling_results.json`, a JSON array of objects with exactly the keys `prompt`, `name` and `parameters`.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install     # uv sync
make run         # uv run python -m src
make debug       # run under pdb
make lint        # flake8 + mypy with the flags required by the subject
make lint-strict # flake8 + mypy --strict
make clean       # remove caches
```

Custom paths:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

or `make run ARGS="--input data/input/un_solo_prompt.json"`.

**Disk space (optional, 42 campus).** The model weights (about 1.5 GB), the virtual environment and the uv cache can exceed a small `$HOME` quota. They are redirected with environment variables, not with `pyproject.toml`, because the paths depend on each user's login:

```bash
export HF_HOME=/sgoinfre/students/<login>/hf_cache
export UV_PROJECT_ENVIRONMENT=/goinfre/<login>/venvs/call_me_maybe
export UV_CACHE_DIR=/goinfre/<login>/uv_cache
```

`goinfre` is a local disk. On another machine the environment has to be recreated with `uv sync`.

**Project layout.**

```
src/
  __main__.py        entry point (python -m src)
  call_me_maybe.py   CLI, main(), per-prompt orchestration
  masking.py         constrained-decoding functions (one per value type)
  tokenizer_utils.py vocabulary loading and id <-> text conversion
  parser.py          pydantic models and input-file loaders
  prompting.py       context sent to the model
  json_utils.py      JSON string escaping
llm_sdk/             provided SDK (local dependency)
```

## Algorithm explanation

**Generation loop.** For each step, `get_logits_from_input_ids(input_ids)` returns one logit per token. The program sets the logits of invalid tokens to `-inf`, takes the `argmax` of what is left, appends that token to `input_ids`, and repeats.

**The JSON is built in segments.** For one prompt, in order:

1. Fixed text: `{"prompt": "<escaped prompt>", "name": "`
2. **Function name** (constrained).
3. Fixed text: `", "parameters": {`
4. For each parameter of the chosen function: the fixed key `"<name>": `, then the **value** (constrained by its declared type), then a fixed separator.
5. Fixed text: `}`

The accumulated text is parsed with `json.loads` as a final check, and `number` values are converted to `float`.

**Constraints by state.**

| State | Allowed tokens |
|---|---|
| Function name | Tokens whose cleaned text is a prefix of the remaining text of some candidate name. Once the generated text equals a full name, the closing quote is also allowed. |
| `number` | ASCII digit tokens. A `,` or `}` is allowed only after at least one digit, and it ends the number. |
| `boolean` | Only the two ids of `true` and `false` (lower case, as JSON requires). |
| `string` | Normal content tokens, plus tokens that close the string. After a backslash, only the JSON escapes `" \ / b f n r t`. Tokens that mix a quote or a backslash with other text are blocked. |

**Token details.**

- The tokenizer marks spaces with `Ġ` and line breaks with `Ċ`. Token text is cleaned before it is compared with real text.
- The model outputs 151936 logits but `vocab.json` has 151643 entries. The 293 extra ids are special tokens with no text. They are computed once (`build_unknown_ids`) and blocked wherever the mask starts from the raw logits.
- A string closes when the model chooses a closing token (`"`, `",`, `"}` ...). That token is not added to the context. The fixed text that follows writes the closing quote and the separator.

**Fixed text.** Fixed segments are tokenized with `model.encode` and appended to the context in one step, without querying the model. The model still sees that text as context for what it generates next.

## Design decisions

- **Masking the whole vocabulary, not a top-N.** I tried sorting the logits and accepting the first valid token among the top 8 and then the top 30. It failed: at the point where the output must start with `{`, that token was at position 134 of the ranking (the top candidates were conversational continuations such as `Please` or `What`). Masking the full vocabulary always finds a valid token, wherever it ranks, and it follows the algorithm described in the subject.
- **Fixed JSON text is written with `encode`, not forced token by token through the model.** The first version forced every character through the model. A full run of the 11 test prompts took 291.6 s and 300.5 s in two runs, at or above the 5-minute limit. Writing the fixed segments with `encode` brought it to 81.5 s. The model still decides the function name and every argument value.
- **Number type.** “Numbers are written as integers. JSON has a single number type, so 2 and 2.0 are both valid values for a number parameter.”
- **Failed prompts.** If a prompt fails, its error is printed on `stderr` and the prompt is left out of the output file, because the subject forbids extra keys in the output. The other prompts are still processed.
- **Output format.** `escribir_resultados` writes each object by hand, passing every value through `json.dumps`, so that `parameters` stays on one line as in the subject's example.
- **Pydantic.** All data classes (`FunctionDefinition`, `ParameterSchema`, `FunctionCallingTest`, `FunctionCallResult`, `FunctionCallError`) use pydantic. Input files are validated, and invalid or missing files stop the program with a clear message.
- **Environment configuration through environment variables.** `uv` cannot set the virtual-environment path from `pyproject.toml`, and a path with a login in it would break the project for the reviewer.
- **Context sent to the model.** The list of all functions is always given, without filtering, because the function must be chosen by the model. The `returns` field is left out of the context, since it does not help to choose a function or its arguments. The context also has an instruction and a one-shot example.

## Performance analysis

**Speed** (one machine, CPU only, 6 cores, 11 prompts):

| Version | Total time |
|---|---|
| Everything forced through the model | 291.6 s and 300.5 s (two runs) |
| Fixed text written with `encode` | 81.5 s |

- A model call costs about 0.57 s with a context of about 200-300 tokens. 96 % of the time is spent inside the model. The Python loops cost about 1.4 s per prompt.
- Time per call depends heavily on the machine. On one campus machine, with PyTorch using all 6 threads and another heavy process running, calls took about 30 s. With 3 threads they took about 5 s, with 4 or 5 about 3.5 s. The solution does not set thread counts, since the reviewer's machine is different.
- Simple prompts take 3-7 s each, and the three replacement prompts take 14-17 s each.

**Reliability.** The structure is guaranteed by the mask: in the final runs all 11 outputs were valid JSON with the correct keys, and the function was the right one in all 11.

**Known limitations.**

- `escape_json_string` handles backslashes and quotes only, not control characters inside the prompt.
- The 60-token string limit would reject a legitimate string longer than that.

## Challenges faced

- **Disk quota.** The model, the virtual environment and the uv cache filled `$HOME` (`No space left on device`). Solved with `HF_HOME`, `UV_PROJECT_ENVIRONMENT` and `UV_CACHE_DIR`.
- **CUDA dependencies.** `uv sync` failed trying to resolve CUDA packages for `torch`. Solved with the CPU-only PyTorch index in `pyproject.toml`.
- **Package layout.** `uv_build` expected `src/call_me_maybe/`. Solved with `module-name = "src"` under `[tool.uv.build-backend]`, so that `python -m src` works.
- **Logits longer than the vocabulary.** A `KeyError` appeared when the model chose one of the 293 ids that have no text. Solved with `build_unknown_ids`.
- **Strings cut after the first token** (`'sh'` instead of `'shrek'`). A `return` was indented inside the `while` loop of `mask_string`.
- **An extra `}` after every parameter.** Two lines were indented inside the parameter loop.
- **Missing quotes around string values.** The string mask drops the closing quote, so the fixed text that follows has to write it.
- **Strings that never closed.** The model wanted to close with tokens such as `",`, which the mask had blocked. Solved by treating any token that starts with a quote and carries no content as a closing token (`es_cierre`).
- **Very slow runs.** One run took 918 s. Printing the time of each call showed about 30 s per call, caused by CPU contention, not by the algorithm.
- **Windows.** `uv sync` failed with `Failed to update Windows PE resources`. Solved with `uv cache clean`.

## Testing strategy

What I know from the work so far:

- Input-file errors were provoked on purpose (missing file, malformed JSON, JSON that does not match the schema), and each one ends with a clear message.
- Each constrained state was run alone against the real model before being combined: boolean, function name, number, string (including a string with escaped quotes) and fixed text.
- The 11 prompts were run, and the output file was loaded with `json.load` and checked for the exact keys `prompt`, `name` and `parameters`.
- Raw logits were inspected at specific steps to understand why the model chose or rejected a token.
- `flake8` and `mypy` with the flags of the subject (`make lint`).
- TODO: say whether there is an automated test suite. If there is none, say so.

## Example usage

Default paths:

```bash
make install
make run
cat data/output/function_calling_results.json
```

Custom input and output:

```bash
uv run python -m src \
    --input data/input/un_solo_prompt.json \
    --output data/output/prueba.json
```

Example output:

```json
[
    {
        "prompt": "What is the sum of 265 and 345?",
        "name": "fn_add_numbers",
        "parameters": {"a": 265, "b": 345}
    },
    {
        "prompt": "Greet shrek",
        "name": "fn_greet",
        "parameters": {"name": "shrek"}
    }
]
```

While running, the program prints one progress line per prompt (`[3/11] ok en 4.1 s`) and the total generation time.

## Resources

- [uv documentation](https://docs.astral.sh/uv/) (projects, build backend, environment variables)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [pydantic documentation](https://docs.pydantic.dev/)
- Peer to Peer

**How AI was used.** TODO: review and adjust this paragraph so that it is exactly true. What I know from the conversation:

- A language model (Claude) was used as a tutor, in a question-and-answer format: to explain how an LLM generates tokens, what logits are and how constrained decoding works, and to plan the design in pseudocode before writing any code.
- It was used to review code that I wrote and to help locate bugs from tracebacks and measurements.
- It was used to look up documentation for `uv` environment variables and the `uv_build` backend, and to draft the `Makefile` and this README.
- TODO: say which parts of the code were written by AI suggestion and which by me, and that all of it was reviewed, tested and can be explained by me.