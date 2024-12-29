from pathlib import Path
from typing import Dict, List
import os
from qgis.core import (
    QgsProject,
    QgsMapLayer,
    QgsFillSymbol,
    QgsSingleSymbolRenderer,
)
from qgis.utils import iface


class Config:
    ROOT_DIR = Path.home() / Path("Pulpit/drony/")
    STYLES = {
        "water_": str(ROOT_DIR / Path("styles/water.qml")),
        "buildings_": str(ROOT_DIR / Path("styles/buildings.qml")),
        "roads_": str(ROOT_DIR / Path("styles/roads.qml")),
        "railways_": str(ROOT_DIR / Path("styles/railways.qml")),
        "landuse_": str(ROOT_DIR / Path("styles/landuse.qml")),
    }
    BUILDINGS_LAYER_NAME = "buildings_"
    GRID_LAYER_NAME = "Siatka"
    GRID_ANALYSIS_LAYER_NAME = "grid_above_5_percent"
    PREFERRED_LAYER_ORDER = [
        "buildings_",
        "roads_",
        "railways_",
        "landuse_",
        "water_",
    ]


class LayerManager:
    @staticmethod
    def get_project() -> QgsProject:
        return QgsProject.instance()

    @staticmethod
    def get_layer_containing(substr: str) -> QgsMapLayer:
        for layer in LayerManager.get_project().mapLayers().values():
            if substr in layer.name():
                return layer
        raise ValueError(f"No layer found containing '{substr}' in name")

    @staticmethod
    def clear_all_selections():
        for layer in LayerManager.get_project().mapLayers().values():
            if layer.type() == QgsMapLayer.VectorLayer:
                layer.removeSelection()

    @staticmethod
    def move_layer_to_top(layer: QgsMapLayer):
        root = LayerManager.get_project().layerTreeRoot()
        node = root.findLayer(layer.id())
        if node:
            clone = node.clone()
            parent = node.parent()
            parent.insertChildNode(0, clone)
            parent.removeChildNode(node)

    @staticmethod
    def reorder_layers(preferred_order: List[str], grid_layer_name: str):
        root = LayerManager.get_project().layerTreeRoot()
        layers = LayerManager.get_project().mapLayers().values()

        vector_layers = [
            layer for layer in layers if layer.type() == QgsMapLayer.VectorLayer
        ]
        raster_layers = [
            layer for layer in layers if layer.type() == QgsMapLayer.RasterLayer
        ]

        ordered_layers = []

        for layer in vector_layers:
            if grid_layer_name.lower() in layer.name().lower():
                ordered_layers.append(layer)

        for layer_name in preferred_order:
            for layer in vector_layers:
                if (
                    layer_name.lower() in layer.name().lower()
                    and layer not in ordered_layers
                ):
                    ordered_layers.append(layer)

        for layer in vector_layers:
            if layer not in ordered_layers:
                ordered_layers.append(layer)

        ordered_layers.extend(raster_layers)

        for layer in reversed(ordered_layers):
            node = root.findLayer(layer.id())
            if node:
                clone = node.clone()
                parent = node.parent()
                parent.insertChildNode(0, clone)
                parent.removeChildNode(node)


class StyleManager:
    @staticmethod
    def apply_styles(styles: Dict[str, str]):
        layers = LayerManager.get_project().mapLayers().values()

        for layer in layers:
            layer_name = layer.name().lower()
            for key, style_path in styles.items():
                if key in layer_name:
                    if os.path.exists(style_path):
                        success = layer.loadNamedStyle(style_path)
                        if success[1]:
                            layer.triggerRepaint()
                            print(f"Applied style for layer: {layer_name}")
                        else:
                            print(f"Failed to load style for layer: {layer_name}")
                    else:
                        print(f"Style file not found: {style_path}")

        iface.mapCanvas().refresh()
        print("Style application completed.")

    @staticmethod
    def set_grid_transparent(grid_layer_name: str):
        layers = LayerManager.get_project().mapLayersByName(grid_layer_name)
        if layers:
            grid_layer = layers[0]
            if grid_layer.type() == QgsMapLayer.VectorLayer:
                symbol = QgsFillSymbol.createSimple(
                    {"fill_style": "no", "color": "0,0,0,0"}
                )
                renderer = QgsSingleSymbolRenderer(symbol)
                grid_layer.setRenderer(renderer)
                grid_layer.triggerRepaint()
