# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Menma1993/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Menma1993/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts each GTA VI PS5 trophy's Exophase earn rate at 2026-11-26T00:00:00Z as five quantiles. It models earners and players_tracked separately in log-time (hours since release) over the case's distinct page states — consecutive identical reads are one state, since Exophase only refreshes its counts every few hours — with slopes taken from the recent half of the states (growth bends as the week goes on: adoption saturates while earner growth steepens), shrunk toward the cross-trophy mean and anchored on the last state. The median is earners / players at the target, aimed half a refresh interval early; the quantiles spread on the logit scale from the fit residual plus a horizon term. When the history is flat (launch night), it falls back to a launch-refresh prior taken from the competition's own published first-refresh counts (players 638 → 3,868, trophies earned 943 → 4,276), with the new earners concentrated on the early story trophies. Node 22, standard library only, no network, deterministic.

## Run it

```sh
MAKEFUN_INPUT_DIR=path/to/input MAKEFUN_OUTPUT_DIR=path/to/output node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=grvy5k3b&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
