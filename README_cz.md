# Generování Světa (Age of Empires styl)

Generování herního světa inspirované Age of Empires 2. Místo generování rovnou na mřížku začínám s grafem regionů - to umožňuje logičtější vztahy (řeky tečou z hor, města jsou blízko zdrojů atd.).

## Princip

Nejdřív graf, pak mřížka. Začínám s Poisson Disc Sampling pro středy regionů, pak Delaunay triangulace pro graf sousednosti. Každému regionu přiřadím hodnoty pomocí Perlin noise a interpoluji je přes KNN. Teprve na konci převedu graf na dlaždice.

## Postup generování

### 1. Poisson Seeds

Poisson Disc Sampling generuje body se minimální vzdáleností - výsledek je přirozenější než náhodné body. Každý bod je středem Voronoi regionu. Parametry: `poisson_radius` určuje hustotu, `poisson_n_points` počet regionů.

### 2. Delaunay Graf

Delaunay triangulace spojí body do trojúhelníkové sítě. Z toho vznikne graf sousednosti - každý region zná své sousedy. Používám scipy.spatial.Delaunay. Graf umožňuje šíření (řeky, lesy) a prohledávání okolí (hledání zdrojů kolem měst).

### 3. Regionální hodnoty + Voronoi

Každému regionu přiřadím elevaci a vlhkost pomocí Perlin noise na různých škálách. Pak interpoluji hodnoty na dlaždice přes KNN s Gaussian váhami (`k_nearest=6`, `sigma=800`). Před finálním určením země/vody aplikuji box blur na elevaci a vlhkost pro plynulejší kontinent. Voronoi diagram přiřadí každou dlaždici k nejbližšímu středu regionu.

Technicky: pro každou dlaždici najdu k nejbližších Poisson bodů, spočítám Gaussian váhy podle vzdálenosti, interpoluji elevaci a vlhkost jako vážený průměr. Perlin FBM přidá detail na různých škálách (octaves=3).

### 4. Řeky

Prameny jsou regiony s vysokou elevací (>0.4) a vlhkostí (>0.3). Řeka teče po hranách grafu vždy k nižšímu sousedovi. Pokud najde souseda s vodou, končí. Šířka řeky je `river_width=5` dlaždic. Pro rasterizaci používám Bresenham-like algoritmus s perpendikulárním offsetem.

### 5. Suroviny

Zlato: hledám regiony s vysokou elevací (>0.5), vytvářím pár malých shluků (1-2 regiony) u startů a jeden větší shluk (4 regiony) daleko od startů. U měst často zlato chybí - pokud je město blízko, zlato se "spotřebuje".

Kámen: jen malé shluky (1 region) u startů a v kopcích (elevace >0.4). Žádný velký cluster.

Dřevo: BFS šíření z seed bodů ve vlhkých oblastech (>wood_threshold). Dřevo se šíří 1 hop do sousedů s podobnou vlhkostí (clustering). Tím vzniknou přírodní hranice lesů.

### 6. Finální rasterizace

Převedu region-level hodnoty na dlaždice. Pro wood: pokud region má wood, dlaždice má ~40-55% šanci být forest (víc u hranic). Pro gold: 15% šance v gold regionu, jinak mountain pokud je vysoká elevace. Pro stone: region se renderuje jako mountain (elevace nebo 40% random). Řeky přepíšou dlaždice na vodu s šířkou river_width.

### 7. Města

Skórování regionů: vážený součet zdrojů v okolí (2 hops BFS). Zlato má váhu 50, stone/wood 10. Město musí mít přístup k wood nebo vodě (2 hops). Hlavní město má nejlepší skóre.

Cesty: hlavní město má cestu do všech ostatních. Plus extra cesty pro města bez zlata - napojí se na město, které ho má. Rasterizace cest: Manhattan path (nejdřív horizontálně, pak vertikálně), nikdy diagonálně. Cesta se nekreslí přes vodu - pokud by měla křížit řeku, segment se přeskočí.

Města rostou organicky BFS od středu, vyhýbají se vodě a horám. City tile má vlastní typ (CITY).

## Spuštění

```bash
python src/maie/main.py
```

Závislosti: pygame, numpy, scipy.

