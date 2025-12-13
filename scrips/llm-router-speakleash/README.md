Aby run_sojka_guardrail.sh działało, trzeba zainstalować bibliotekę `llm-router-services`:

```bash
pip install git+https://github.com/radlab-dev-group/llm-router-services.git
```

Dowiązujemy pobrany model `speakleash/Bielik-Guard-0.1B-v1.0` do lokalizacji, gdzie uruchamiamy model,
tam gdzie leży skrypt uruchamiający `run-sojka-guardrail.sh`

```bash
ln -s /mnt/data2/llms/models/community/speakleash/Bielik-Guard-0.1B-v1.0/ speakleash/
```

**UWAGA** `run-sojka-guardrail` uruchamia dwa workery gunicorna z modelem ładowanym na `cuda:0`.