# Ассоциации к простым словам на двух битах

2026-09-16, gemma-4-E2B-it. Разведка E017 до предрегистрации: двенадцать простых слов, по каждому просьба «Write five words you
associate with "<слово>". Only the words, separated by commas.», жадная генерация, до 40 токенов, первая строка
ответа. Контролируемые веса заменены раскрытой копией и читаются в bf16.

| Слово | Наш D2 | Q2_K bartowski | UD-Q2_K_XL unsloth |
| --- | --- | --- | --- |
| cat | Highest: | purr, sleek, curious, playful, soft | Feline, purr, sleek, playful, soft |
| dog | ... | Loyal, playful, wagging, bark, canine | Loyal, playful, energetic, furry, companion |
| house | `** $\text{...**$ ** $\quad**` | brick, roof, walls, door, home | Roof, walls, door, structure, home |
| water | sings. | liquid, flow, ocean, dew, wet | Liquid, clear, flow, wet, life |
| sun | `**Delta** ** ** ****` | bright, yellow, warm, golden, light | Bright, warm, radiant, golden, light |
| tree | `Structure 望im... �nong...` | wood, trunk, roots, branches, forest | Wood, roots, branches, leaves, trunk |
| bread | `Summer is tallest... $\s$` | wheat, crust, sourdough, yeast, grain | Yeast, grain, crust, baked, wholesome |
| book | `ed. 望a. / / / /` | story, pages, chapters, novel, read | read, story, pages, knowledge, imagination |
| doctor | ... | stethoscope, diagnosis, patient, doctor, medical | Healer, physician, diagnosis, expertise, medical. |
| winter | `Shares 望gruouslys ... ** **` | snow, ice, frost, cold, frosty | Snowy, cold, ice, frost, white |
| кошка | `$sings$sings $\textcposal$` | cat, purr, sleek, graceful, feline | Feline, graceful, sleek, mysterious, playful. |
| собака | `BLIM / / ... *s$` | bark, loyal, playful, energetic, canine | Loyal, playful, energetic, obedient, canine. |

**Наш D2 испортил модель.** На всех двенадцати словах - обрывки разметки, символы и иероглифы, ни одной
ассоциации. Готовые двухбитные файлы того же чекпоинта отвечают осмысленно на все двенадцать, по-русски тоже
(ассоциации - по-английски).

**Бюджет почти тот же, механика другая.** Наш D2 квантует только контролируемые модули (внимание, MLP,
per-layer), эмбеддинги остаются в bf16. На тех же модулях GGUF держит в основном 2-3 бита:

| | 2 бита | 3 бита | 4 бита и выше |
| --- | --- | --- | --- |
| Q2_K bartowski | 61% (Q2_K) | 35% (Q3_K) | 4% |
| UD-Q2_K_XL unsloth | 55% (Q2_K, IQ2_S, IQ2_XS) | 39% (Q3_K, IQ3_S, IQ3_XXS) | 7% |

По всем весам файла доля двух бит меньше (25% и 22%): эмбеддинги в GGUF на Q6_K и Q4_K. Наш D2 - все
контролируемые веса на двух битах: симметричный шаг amax/2 на группу 64, без нуля и без подбора шага; Q2_K -
шкала и минимум на блок из 16 с подбором, сами шкалы квантованы на супер-блок из 256. Модель на таком
бюджете можно не ломать.

**Что из этого следует.** Механику воздействия на сеть надо пересмотреть. Если наша механика ломает D2, она
может частично повреждать и D4: тогда потеря D4 в E016 (14.3% знаний bf16) меряет наш способ квантования.
Это косвенный признак. Следующий шаг - наш D4 против Q4_K того же чекпоинта на
тех же вопросах. Зонд классов модулей ([разведка классов модулей](exploration-module-classes.ru.md))
добавляет оговорку: асимметричная шкала с подбором на группе 16, все веса на двух битах, учит D2
останавливаться, но знаний не возвращает (EM 0.048) - на двух битах решает и механика, и то, какие классы
держать выше.
