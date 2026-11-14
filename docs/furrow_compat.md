# Совместимость с Furrow

Статус: **не совместимо**. Наш экспорт (`contracts/schema.json`, модель `FieldAnalysis`) не совпадает с форматом, который Furrow читает из `data/ml/<region>.json`. Ниже — расхождения и обязательность полей.

Источники, только чтение:
- `D:\carver-data\furrow\web\src\types.ts` — тип `Region.ml` (строка 69)
- `D:\carver-data\furrow\data\ml\README.md` — контракт с ML-репозиторием

## Что ожидает Furrow

```ts
ml: null | {
  model: string
  version: string
  validation: string
  cells: Record<string, { crop?: string; crop_p?: number; sm_rootzone?: number }>
}
```

- Ключи `cells` — H3-ячейки разрешения 5 (`pipeline/regions.py`).
- Файл кладётся в `data/ml/<region>.json`, регионы `az` или `vko`. Если файла нет, слоя ML не будет и ничего не сломается.
- `validation` показывается в приложении как есть.

## Что экспортирует наш репо

`FieldAnalysis` (`src/fs_ml/contracts.py`):

- `field_id`, `region`, `model_version`
- `crop_history`: список `{year, crop, prob}`
- `bare_soil`: список `{year, days, ci_low, ci_high}`
- `domain_shift`, `data_gaps`

Один объект описывает **одно поле**. Никакой H3-привязки, никакого поля `cells`, `validation` или `model` в нём нет. В `pipeline/` или `src/` H3 нигде не используется.

## Расхождения

| Что | Furrow | Наш экспорт | Вывод |
|---|---|---|---|
| Гранулярность | Ячейка H3 res 5 | Поле (`field_id`) | Нужна агрегация пикселей в H3 res 5. Сейчас её нет. |
| Ключ | H3-строка в `cells` | `field_id` | Не совпадает. |
| Культура | `cells[h3].crop` (строка) | `crop_history[].crop` по годам | Структура другая: у нас история по годам, у них одно значение на ячейку. |
| Вероятность | `crop_p` | `prob` | Имя другое. |
| Влажность почвы | `sm_rootzone`, `sm_rootzone_pi` | нет | Не наш продукт, см. ниже. |
| Метаданные модели | `model`, `version`, `validation` на верхнем уровне | `model_version` внутри каждого поля | Отсутствуют `model` и `validation`, `version` есть только как `model_version`. |
| Лишние поля | — | `bare_soil`, `crop_history`, `domain_shift`, `data_gaps` | Furrow их не читает. Ничего не ломают, но и не используются. |

## Обязательность полей со стороны Furrow

Верхний уровень (по типу `ml`, не `optional`):
- `model`, `version`, `validation`, `cells` — обязательны. Без них файл не соответствует типу.

Внутри `cells`:
- `crop`, `crop_p`, `sm_rootzone` — все необязательны (README: «every field per cell is optional»).
- `sm_rootzone_pi` — есть в README, но **в `types.ts` не описан**. Если добавим, TypeScript его проигнорирует, но тип не будет его отражать.

Наш экспорт: ни одного из этих полей на верхнем уровне нет, поэтому файл не пройдёт проверку по типу.

## Что нужно сделать, чтобы совпасть

1. Агрегировать пиксельные результаты в ячейки H3 res 5 внутри bbox.
2. Записать файл в формате `{model, version, validation, cells}` с ключами H3.
3. В `cells` заполнять `crop` и `crop_p` (доля пикселей ячейки с доминирующей группой и её средняя вероятность). `sm_rootzone` не трогаем: это не наш выход.
4. Проверить, что значения `crop` совпадают с ключами `crops` в `Region` (`types.ts`, строка 62). Сейчас у нас группы «alfalfa / cotton / small grains / corn / fallow / other», у Furrow id вроде `wheat`, `fallow`, `cotton`. Для нашего региона таблица `crops` в данных Furrow не сверялась, это отдельная проверка.
5. Поле `validation` заполнить честно и по нашей модели. Пример в README Furrow («F1 0.81 on held-out Kansas counties») относится к другой модели и другому региону. Подставлять его нельзя. Пока нашего F1 нет, поле останется заглушкой до baseline.
6. Версию в `version` ставить явно. В примере стоит `2026-11-15`, а сегодня 2026-11-14, поэтому пример, скорее всего, просто дата-заглушка.

## Что не проверено

- Не запускали `pipeline/build.py` Furrow и не проверяли, как он читает файл на практике. Сравнение сделано по типам и README.
- Не проверяли, есть ли в `data/ml/` уже файлы. Судя по README, это место для будущих файлов ML-репо.

## Ключи культур: наши группы CDL против Furrow

Источник ключей: `D:\carver-data\furrow\pipeline\crops.py` (`CROPS`), подтверждено `web/public/data/az.json` и `vko.json`. В `types.ts` ключей нет (там `Record<string, …>`), в `data/ml/README.md` только пример `wheat`.

Ключи Furrow для AZ: `cotton`, `durum`, `fallow`, `guar`, `sorghum`, `cover`, `barley`.
Ключи Furrow для VKO: `wheat`, `fallow`, `lentil`, `pea`, `flax`, `rapeseed`, `sainfoin`, `chickpea`, `millet`.

| Наша группа | Коды CDL | Ключ в Furrow AZ | Ключ в Furrow VKO | Вывод |
|---|---|---|---|---|
| alfalfa (люцерна) | 36 | нет | нет | Нет ключа. Ближайший по роли — `sainfoin` (многолетний корм), но он только в VKO. В AZ ближайшего корма нет, нужен новый ключ либо маппинг в `cover`/другую группу. |
| cotton (хлопок) | 2 | `cotton` | нет | Совпадает. |
| мелкие зерновые | 22, 21 | `barley` (ячмень, код 21); `durum` (твёрдая пшеница, близко к 22) | `wheat` (только VKO) | Частично. Ячмень совпадает. Пшеницы в AZ нет, ближайший — `durum`. Наша группа объединяет пшеницу и ячмень, у Furrow это разные ключи. |
| corn (кукуруза) | 1 | нет | нет | Нет ключа. Ближайший по сезону и семейству злаков — `sorghum` (зерновое сорго). Это другая культура, маппинг нельзя делать молча. |
| fallow/idle | 61 | `fallow` | `fallow` | Совпадает по ключу. Значения разные: в AZ «Summer fallow», в VKO «Black fallow», оба с нулевой урожайностью. |

Дополнительно:
- `wheat` в `data/ml/README.md` существует только в VKO. Если ML-файл для AZ будет содержать `wheat`, Furrow его не сопоставит с ключами региона.
- Если экспорт идёт по нашим группам, а не по ключам Furrow, в `crop` надо писать ключи Furrow. Иначе подписи в приложении и сравнение с `crops` не сработают.
- Решение о маппинге (особенно `alfalfa` и `corn`, где ближайшие ключи — другие культуры) принимает команда, это не техническая правка.
