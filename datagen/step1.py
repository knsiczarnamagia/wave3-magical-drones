import os
import uuid
from pathlib import Path
from qgis.core import (
    QgsProject,
    QgsMapLayer,
    QgsVectorLayer,
    QgsFeature,
    QgsField,
    QgsSpatialIndex,
    # QgsCoordinateReferenceSystem
)
# from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtCore import QSize, QVariant, Qt
from PyQt5.QtWidgets import QProgressDialog
from qgis.utils import iface
from qgis import processing

ROOT_DIR = Path.home()/Path("Pulpit/drony/")
STYLES = {
    "water_": str(ROOT_DIR/Path("styles/water.qml")),
    "buildings_": str(ROOT_DIR/Path("styles/buildings.qml")),
    "roads_": str(ROOT_DIR/Path("styles/roads.qml")),
    "railways_": str(ROOT_DIR/Path("styles/railways.qml")),
    "landuse_": str(ROOT_DIR/Path("styles/landuse.qml")),
}
BUILDINGS_LAYER_NAME = "buildings_"
GRID_LAYER_NAME = "Siatka"


def reproject_layers_to_2180():
    """Konwertuje wszystkie warstwy wektorowe z EPSG:4326 do EPSG:2180"""
    project = QgsProject.instance()
    layers = project.mapLayers().values()
    
    # Progress dialog
    vector_layers = [layer for layer in layers if isinstance(layer, QgsVectorLayer)]
    progress = QProgressDialog("Reprojekcja warstw...", "Anuluj", 0, len(vector_layers))
    progress.setWindowModality(Qt.WindowModal)
    progress.show()
    
    current = 0
    
    for layer in vector_layers:
        if progress.wasCanceled():
            break
            
        source_crs = layer.crs()
        if source_crs.authid() == 'EPSG:4326':
            # Utworzenie nazwy dla nowej warstwy
            new_layer_name = f"{layer.name()}_2180"
            
            # Utworzenie parametrów dla reprojekcji
            params = {
                'INPUT': layer,
                'TARGET_CRS': 'EPSG:2180',
                'OUTPUT': 'memory:' + new_layer_name
            }
            
            # Wykonanie reprojekcji
            result = processing.run("native:reprojectlayer", params)
            
            # Dodanie nowej warstwy do projektu
            new_layer = result['OUTPUT']
            project.addMapLayer(new_layer)
            
            # Usunięcie starej warstwy
            project.removeMapLayer(layer)
            
            print(f"Przekonwertowano warstwę: {new_layer.name()} do EPSG:2180")
        
        current += 1
        progress.setValue(current)
    
    progress.close()
    iface.mapCanvas().refresh()
    print("Zakończono reprojekcję warstw.")

def reorder_layers(preferred_order, grid_layer_name):
    """Zmiana kolejności warstw w projekcie"""
    root = QgsProject.instance().layerTreeRoot()
    layers = QgsProject.instance().mapLayers().values()
    
    vector_layers = [layer for layer in layers if layer.type() == QgsMapLayer.VectorLayer]
    raster_layers = [layer for layer in layers if layer.type() == QgsMapLayer.RasterLayer]
    
    ordered_layers = []
    
    # Dodawanie warstw w określonej kolejności
    for layer in vector_layers:
        if grid_layer_name.lower() in layer.name().lower():
            ordered_layers.append(layer)
            
    for layer_name in preferred_order:
        for layer in vector_layers:
            if layer_name.lower() in layer.name().lower() and layer not in ordered_layers:
                ordered_layers.append(layer)

    for layer in vector_layers:
        if layer not in ordered_layers:
            ordered_layers.append(layer)

    ordered_layers.extend(raster_layers)

    # Aktualizacja kolejności w drzewie warstw
    for layer in reversed(ordered_layers):
        node = root.findLayer(layer.id())
        if node:
            clone = node.clone()
            parent = node.parent()
            parent.insertChildNode(0, clone)
            parent.removeChildNode(node)

def apply_styles(styles):
    """Aplikacja stylów do warstw"""
    layers = QgsProject.instance().mapLayers().values()
    
    for layer in layers:
        layer_name = layer.name().lower()
        for key, style_path in styles.items():
            if key in layer_name:
                if os.path.exists(style_path):
                    success = layer.loadNamedStyle(style_path)
                    if success[1]:
                        layer.triggerRepaint()
                        print(f"Zastosowano styl dla warstwy: {layer_name}")
                    else:
                        print(f"Nie udało się załadować stylu dla warstwy: {layer_name}")
                else:
                    print(f"Nie znaleziono pliku stylu: {style_path}")
    
    iface.mapCanvas().refresh()
    print("Zakończono aplikację stylów.")

def get_layer_containing(substr):
    layer = None
    for layer in QgsProject.instance().mapLayers().values():
        if substr in layer.name():
            layer = layer
            break
    if layer is None:
        raise ValueError(f"Nie znaleziono warstwy zawierającej '{layer}' w nazwie")
    return layer

def move_layer_to_top(layer):
    """Przenosi warstwę na samą górę w panelu warstw"""
    root = QgsProject.instance().layerTreeRoot()
    node = root.findLayer(layer.id())
    if node:
        clone = node.clone()
        parent = node.parent()
        parent.insertChildNode(0, clone)
        parent.removeChildNode(node)

def analyze_grid():
    """Analiza siatki i tworzenie nowej warstwy z oczkami zawierającymi >5% budynków"""
    # Wczytaj warstwy
    buildings_layer = get_layer_containing(BUILDINGS_LAYER_NAME)
    grid_layer = QgsProject.instance().mapLayersByName("Siatka")[0]

    # Utwórz spatial index dla budynków
    building_index = QgsSpatialIndex()
    for feat in buildings_layer.getFeatures():
        building_index.addFeature(feat)

    # Utwórz nową warstwę
    result_layer = QgsVectorLayer("Polygon?crs=" + grid_layer.crs().toWkt(), "grid_above_5_percent", "memory")
    result_layer_provider = result_layer.dataProvider()

    # Dodaj pola
    result_layer_provider.addAttributes(grid_layer.fields())
    result_layer_provider.addAttributes([QgsField("building_percent", QVariant.Double)])
    result_layer.updateFields()

    # Utworzenie progress dialog
    feature_count = grid_layer.featureCount()
    progress = QProgressDialog("Analizowanie siatki...", "Anuluj", 0, feature_count)
    progress.setWindowModality(Qt.WindowModal)
    progress.show()

    # Przygotuj features do dodania
    features_to_add = []
    current = 0

    # Główna pętla
    for grid_feature in grid_layer.getFeatures():
        if progress.wasCanceled():
            break
            
        grid_geom = grid_feature.geometry()
        grid_area = grid_geom.area()
        
        if grid_area == 0:
            continue
        
        # Znajdź potencjalne przecięcia używając spatial index
        intersecting_ids = building_index.intersects(grid_geom.boundingBox())
        building_area_total = 0
        
        # Sprawdź tylko budynki, które mogą się przecinać
        for building_id in intersecting_ids:
            building_feature = buildings_layer.getFeature(building_id)
            building_geom = building_feature.geometry()
            
            if grid_geom.intersects(building_geom):
                intersection = grid_geom.intersection(building_geom)
                if not intersection.isEmpty():
                    building_area_total += intersection.area()
        
        building_percent = (building_area_total / grid_area) * 100
        
        if building_percent > 5:
            new_feature = QgsFeature(result_layer.fields())
            new_feature.setGeometry(grid_geom)
            new_feature.setAttributes(grid_feature.attributes() + [building_percent])
            features_to_add.append(new_feature)
        
        # Aktualizuj progress bar
        current += 1
        progress.setValue(current)

    # Dodaj wszystkie features za jednym razem
    result_layer_provider.addFeatures(features_to_add)

    # Zamknij progress dialog
    progress.close()

    # Dodaj warstwę do projektu
    QgsProject.instance().addMapLayer(result_layer)

    # Odśwież widok mapy
    move_layer_to_top(result_layer)
    iface.mapCanvas().refresh()


def main():    
    preferred_order = [
        "buildings_",
        "roads_",
        "railways_",
        "landuse_",
        "water_",
    ]
    # Zmień układ współrzędnych
    reproject_layers_to_2180()
    
    # Zmiana kolejności warstw
    print("Zmiana kolejności warstw...")
    reorder_layers(preferred_order, GRID_LAYER_NAME)
    
    # Aplikacja stylów
    print("Aplikowanie stylów...")
    apply_styles(STYLES)
    
    # Odświeżenie widoku
    iface.mapCanvas().refresh()
    
    # Analiza siatki
    print("Rozpoczynanie analizy siatki...")
    analyze_grid()
    
    print("Proces zakończony pomyślnie!")


main()