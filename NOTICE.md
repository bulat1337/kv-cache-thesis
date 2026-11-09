# Источники данных и сторонние компоненты

## HotpotQA

В `research/pilot/data/examples.jsonl` содержится небольшая выборка HotpotQA вместе с синтетическими примерами. Тексты документов, естественные вопросы, ответы и supporting facts взяты из HotpotQA и распространяются на условиях [Creative Commons Attribution–ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/).

Авторы HotpotQA: Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, Christopher D. Manning.

> Yang et al. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. EMNLP, 2018.

[Официальная страница](https://hotpotqa.github.io/), [публикация](https://aclanthology.org/D18-1259/), [зеркало данных](https://huggingface.co/datasets/hotpotqa/hotpot_qa/tree/1908d6afbbead072334abe2965f91bd2709910ab).

Преобразования: детерминированный отбор восьми исходных вопросов с полным запросом до 2048 токенов; сохранение всех десяти документов; циклическая перестановка документов для второго варианта; объединение предложений абзаца; добавление шаблона запроса и token IDs. Тексты фактов и эталонные ответы не редактировались. Производная выборка HotpotQA сохраняет условия CC BY-SA 4.0. Синтетические записи обозначены полем `dataset: synthetic`.

## Модель и библиотеки

Веса модели и исходные деревья сторонних библиотек загружаются отдельно и сохраняют собственные лицензии:

| Компонент | Версия | Лицензия |
|---|---|---|
| Qwen3-1.7B | revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | [Apache-2.0](https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/LICENSE) |
| MLX | 0.32.4 | [MIT](https://github.com/ml-explore/mlx/blob/36969c094354086e15d2a327a9c50feb20b9cf11/LICENSE) |
| MLX-LM | 0.32.2 | [MIT](https://github.com/ml-explore/mlx-lm/blob/93e6d4a6447b3080861985947f0820e76fcb957d/LICENSE) |
| EPIC, сравнительный исходный прототип | commit `3204410f1723ed0f39575b4eba8c17578a1ee1a1` | [Apache-2.0](https://github.com/DerekHJH/epic/blob/3204410f1723ed0f39575b4eba8c17578a1ee1a1/LICENSE) |

Официальный [scorer HotpotQA](https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/hotpot_evaluate_v1.py) загружается из исходного проекта для независимой проверки метрик; его commit и SHA256 закреплены. Копия scorer в репозитории не распространяется. Лицензия данных HotpotQA относится к данным и не подменяет условия использования стороннего кода.

Лицензии сторонних материалов сохраняют свой исходный объём и не распространяются автоматически на остальные файлы исследования.
