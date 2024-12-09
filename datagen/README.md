# How to use this "pipeline"?

1. Download and import vector layers to your project
    https://download.geofabrik.de/europe/poland.html (.shp.zip files)
    Used layers: buildings, roads, railways, water, landuse

2. Create grid for area of your interests (not too big, e.g. one city) in QGIS (Panel algorytmów -> Utwórz siatkę)
    - I recommend selecting some area of buildings layer and creating new layer from it (Eksport -> zapisz wybrane obiekty jako...) and then making grid based on this new layer.

3. Run `step1.py` script. It will set styles for vector layers (from `styles` dir to maintain consistency), reorder them, convert layers to 2180 coord system and create new grid with only selected cells that have >5% of buildings in its area (new layer with name `grid_above_5_percent`).
    - `landuse.qml` style file will show only 'forest' (green) regions and hide all others

4. Download orto maps for `grid_above_5_percent` layer.

5. Run `step2.py` script. It will save vector and orto tiles to `/zdjecia` and `/mapy` directories.
    - Remember to deselect everything, as some selected buildings may turn yellow instead of red when extracted.

6. If you are done with this vector layer (Voivodeship) you can run `step3.py` to clear your project (remove all layers and deletes all orto files)

## Remember to adjust your paths in scrpits!