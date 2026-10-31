# Примеры запроса и ответа `POST /triage`

* `triage_request.json` — минимальный запрос из §8.1 описания кейса (жалоба на оскорбление в чате,
  игрок сам указал пункт правил `2.1 ОП`, 2 вложения).
* `triage_response.json` — реальный ответ сервиса на этот запрос (получен на `data/rules.json` из handout;
  на синтетическом fixture из `tests/fixtures/rules.json` структура ответа та же, значения полей могут отличаться).

Проверка на запущенном сервисе (macOS / Linux):

```bash
curl -s -X POST http://127.0.0.1:8000/triage \
  -H 'Content-Type: application/json' \
  --data @examples/triage_request.json | python3 -m json.tool
```

Windows PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/triage `
  -ContentType "application/json; charset=utf-8" `
  -Body (Get-Content examples\triage_request.json -Raw) |
  ConvertTo-Json -Depth 8
```

Ожидаемые ключевые поля ответа: `lang = "ru"`, `category = "complaint.insult"`, `priority = "p2"`,
`extracted.nickname = "Arman_Petrov"`, `extracted.has_evidence = true`, `draft_reply.cited = ["2.1"]`,
`needs_human = true`, `confidence = 0.93`.
