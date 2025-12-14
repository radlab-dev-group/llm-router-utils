# llm‑router‑utils

## What is it?

`llm-router-utils` is a collection of ready‑made **tools and examples** built on top of **LLM‑Router** – a flexible
language‑model router. The repository contains:

* **Ready‑to‑run examples** – scripts, configurations, and small applications that you can simply clone and execute.
* **Universal tools** – useful in many areas (data analysis, text processing, content protection, etc.).
* **Minimal setup** – everything you need is in the repository; after cloning, just run a few commands.

## Why use it?

* **Fast start** – you don’t have to build infrastructure from scratch; all components are already prepared.
* **Modularity** – you can pick only the parts you need and easily combine them.
* **Extensibility** – thanks to LLM‑Router you can plug in any model (local, cloud‑based, custom service) and take
  advantage of built‑in load‑balancing strategies.

## What’s in the repository?

| Directory / File                                                    | Description                                                                                                       |
|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| [resources/llm-router-speakleash/](resources/llm-router-speakleash) | Example configuration and launch scripts for Speakleash models (e.g., Bielik‑11B‑v2.3‑Instruct and Bielik‑Guard). |
| `run‑*.sh`                                                          | Bash start‑up scripts (VLLM, REST‑API, Guardrail). Just make them executable (`chmod +x`) and run them.           |
| `llm_router_utils/`                                                 | Python package with helper functions (currently empty, ready for extension).                                      |
| `README.md` (this file)                                             | Guide to the repository.                                                                                          |
| `requirements.txt` (optional)                                       | List of dependencies, if you decide to add your own libraries.                                                    |

## Summary

`llm-router-utils` is a **compact toolkit** that lets you test and deploy LLM‑Router‑based solutions within minutes.
Just **clone, set permissions, and run** – and you’ll have a working content‑generation and protection pipeline.

Happy coding! 🚀