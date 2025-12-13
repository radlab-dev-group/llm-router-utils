Aby run_sojka_guardrail.sh działało, trzeba zainstalować bibliotekę `llm-router-services`:

```bash
pip install git+https://github.com/radlab-dev-group/llm-router-services.git
```

Dowiązujemy pobrany model `speakleash/Bielik-Guard-0.1B-v1.0` do lokalizacji, gdzie uruchamiamy model,
tam gdzie leży skrypt uruchamiający `run-sojka-guardrail.sh`. Tam gdzie odpalany jest skrypt,
wymagane jest aby model leżał pod ściezką `ls speakleash/Bielik-Guard-0.1B-v1.0`:

```bash
ln -s /mnt/data2/llms/models/community/speakleash/Bielik-Guard-0.1B-v1.0/ speakleash/
```

Wtedy:
```bash
segfault:community $ ls -la speakleash/Bielik-Guard-0.1B-v1.0
razem 492356
drwxrwxr-x 2 pkedzia pkedzia      4096 lis 24 12:55 .
drwxrwxr-x 6 pkedzia pkedzia      4096 gru 13 12:37 ..
-rw-rw-r-- 1 pkedzia pkedzia       948 lis 24 12:53 config.json
-rw-rw-r-- 1 pkedzia pkedzia      1519 lis 24 12:53 gitattributes
-rw-rw-r-- 1 pkedzia pkedzia 497811044 lis 24 12:54 model.safetensors
-rw-rw-r-- 1 pkedzia pkedzia     15577 lis 24 12:53 README.md
-rw-rw-r-- 1 pkedzia pkedzia       964 lis 24 12:53 special_tokens_map.json
-rw-rw-r-- 1 pkedzia pkedzia      1468 lis 24 12:53 tokenizer_config.json
-rw-rw-r-- 1 pkedzia pkedzia   3355765 lis 24 12:53 tokenizer.json
-rw-rw-r-- 1 pkedzia pkedzia   2953979 lis 24 12:53 unigram.json
```

**UWAGA** `run-sojka-guardrail` uruchamia dwa workery gunicorna z modelem ładowanym na `cuda:0`.