# E006: перефразы при допуске оригинала

Те же пять шкал на тех же 50 перефразах, что в `fifty-paraphrased`, но поля пересчитаны при допуске их
оригиналов: вопрос и ответ те же, значит и разрешённая потеря должна быть той же
(`precision_fields --tolerance-from`).

Серия ставилась, чтобы проверить объяснение проигрыша карт на перефразах: допуск задан как
`max(0.03 ната, 0.2 от потерь полной модели)`, а на перефразе модель менее уверена, поэтому допуск и порог
раздуваются, а карта беднеет.

## Объяснение не подтвердилось

| раскладка | свой допуск | допуск оригинала |
| --- | --- | --- |
| error_energy | 26 из 50, трудных 9 из 25, цена 0.587 | 27 из 49, трудных 11 из 24, цена 0.674 |
| lift_per_weight | 22 из 43, трудных 5 из 23, вырожденных 7 | 22 из 40, трудных 5 из 21, вырожденных 10 |
| ровная D6 | 33 из 50, трудных 16 из 25, цена 0.775 | то же |

Энергия ошибки прибавила двух трудных и подорожала на 0.09. У `lift_per_weight` изменилось только число
вырожденных карт: при меньшем допуске порог чаще не находится вовсе, и вся сеть уезжает наверх.

**Ровная D6 остаётся впереди всех карт и при честном допуске.** Причина проигрыша другая: перефразы легче для
ровной ступени - она берёт на них 66% против 44% на оригиналах, - и карте нечего отыгрывать.

Допуск при этом всё равно задан неудачно: выделять меньше точности вопросу, который модель знает хуже,
неправильно по смыслу. Но проигрыш карт на перефразах им не объясняется.

## Как повторить

    uv run python scripts/precision_fields.py --config experiments/E006-filter-map-retention/configs/precision-fields-paraphrased.toml \
        --wordings prompts/paraphrases --only ../sample/paraphrased.json --also ../sample/all.json \
        --tolerance-from ../precision-fields-bartowski-Q2_K-small-corpus.npz --out ../paraphrased-fair
    uv run python scripts/build_maps.py --fields ../paraphrased-fair/precision-fields-*.npz --out ../paraphrased-fair/maps
    uv run python scripts/band_grid.py --maps ../paraphrased-fair/maps/maps-*.npz \
        --fields ../paraphrased-fair/precision-fields-*.npz --masks ../paraphrased/masks/e2b-it/error_energy-*.npz \
        --out . --bands 0.50,0.95,1.00 0.50,0.80,1.00 0.35,0.60,0.99 0.35,0.60,0.88 0.50,0.667,0.95
    uv run python scripts/oracle_answers.py --config experiments/E006-filter-map-retention/configs/answers-paraphrased-fair.toml \
        --base bartowski-Q2_K --out . --gpu-share 0.9 --also ../sample/all.json --wordings prompts/paraphrases
