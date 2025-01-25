# Dataset Card for mag-map

## Table of Contents

- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-instances)
  - [Data Splits](#data-instances)
- [Dataset Creation](#dataset-creation)
  -[Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)


## Dataset Description

-  **Homepage:** https://huggingface.co/datasets/czarna-magia/mag-map

-  **Repository:** https://github.com/knsiczarnamagia/wave3-magical-drones

-  **Point of Contact:** [Czarna Magia Discord](https://discord.com/invite/aVTeDfreSD)

### Dataset Summary

This dataset consists of ~16,000 pairs (32,000 total files) of satellite images and their corresponding vector map images of Polish cities. Each file is named using the structure: `{city}_cell_{number}.jpg` (e.g., `warszawa_cell_1.jpg`).

The dataset was created using QGIS software and the following resources:
- Vector maps sourced from the [Geofabrik portal](https://download.geofabrik.de/europe.html),
- The [GUGiK Data Downloader plugin for QGIS](https://plugins.qgis.org/plugins/pobieracz_danych_gugik/),
- Custom scripts available in [project's repository](https://github.com/knsiczarnamagia/wave3-magical-drones).

This collection is suitable for applications in geographic information systems, urban analysis, and machine learning tasks involving spatial data.

This dataset card was created to fulfill one of the WAVE3 obligatory project requirements, as the dataset is part of the Czarna Magia Student Artificial Intelligence Society's project.


## Dataset Structure

### Data Instances
```javascript  
{
  sat_image: "localhost:8080/sat/warszawa_cell_1.jpg",
  map_image: "localhost:8080/map/warszawa_cell_1.jpg",
  name: "warszawa_cell_1.jpg"
}
```
  
### Data Fields 

- sat_image: a `.jpg` image
- map_image: a `.jpg` image
- name: a string

## Dataset Creation
  
### Initial Data Collection and Normalization

The data was gathered through the following steps:

1. Import shapefile vectors of Polish voivodeships from the [Geofabrik portal].(https://download.geofabrik.de/europe.html) into [QGIS](https://www.qgis.org)
2. Create a grid overlay on the selected city to prevent crashes during processing.
3. Use the first script from [our repository](https://github.com/knsiczarnamagia/wave3-magical-drones) at `./datagen/step1.py` to:
    - Standardize styles, coordinate system, and layers.
    - Create a new grid containing only cells over terrain with more than 5% building layer coverage.
4. Download the satellite images using the [GUGiK Data Downloader plugin for QGIS](https://plugins.qgis.org/plugins/pobieracz_danych_gugik/).
5. Import the satellite images from the directory chosen in the previous step to the program.
6. Use the second script from [our repository](https://github.com/knsiczarnamagia/wave3-magical-drones) at `./datagen/step2.py` to crop the vector and satellite files according to grid cells and export them to the ./sat and ./map directories with appropriate file names.

## Additional Information

### Dataset Curators

- Dawid Koterwas
- Tarik Alaiwi

### Licensing Information


MIT License

Copyright (c) [2025] [KNSI Czarna Magia]


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
