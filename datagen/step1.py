from utils import Config, LayerManager, StyleManager
from qgis import processing
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QProgressDialog
from qgis.utils import iface
from PyQt5.QtCore import QVariant
from qgis.core import QgsVectorLayer, QgsSpatialIndex, QgsFeature, QgsField


class GridAnalyzer:
    def __init__(self):
        self.buildings_layer = LayerManager.get_layer_containing(
            Config.BUILDINGS_LAYER_NAME
        )
        self.grid_layer = LayerManager.get_project().mapLayersByName("Siatka")[0]

    def reproject_layers_to_2180(self):
        layers = LayerManager.get_project().mapLayers().values()
        vector_layers = [layer for layer in layers if isinstance(layer, QgsVectorLayer)]

        progress = QProgressDialog(
            "Reprojecting layers...", "Cancel", 0, len(vector_layers)
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        for idx, layer in enumerate(vector_layers):
            if progress.wasCanceled():
                break

            if layer.crs().authid() == "EPSG:4326":
                new_layer_name = f"{layer.name()}_2180"
                params = {
                    "INPUT": layer,
                    "TARGET_CRS": "EPSG:2180",
                    "OUTPUT": "memory:" + new_layer_name,
                }
                result = processing.run("native:reprojectlayer", params)
                new_layer = result["OUTPUT"]
                LayerManager.get_project().addMapLayer(new_layer)
                LayerManager.get_project().removeMapLayer(layer)

            progress.setValue(idx + 1)

        progress.close()
        iface.mapCanvas().refresh()

    def analyze_grid(self):
        building_index = QgsSpatialIndex()
        for feat in self.buildings_layer.getFeatures():
            building_index.addFeature(feat)

        result_layer = QgsVectorLayer(
            "Polygon?crs=" + self.grid_layer.crs().toWkt(),
            Config.GRID_ANALYSIS_LAYER_NAME,
            "memory",
        )
        provider = result_layer.dataProvider()
        provider.addAttributes(self.grid_layer.fields())
        provider.addAttributes([QgsField("building_percent", QVariant.Double)])
        result_layer.updateFields()

        progress = QProgressDialog(
            "Analyzing grid...", "Cancel", 0, self.grid_layer.featureCount()
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        features_to_add = []
        for idx, grid_feature in enumerate(self.grid_layer.getFeatures()):
            if progress.wasCanceled():
                break

            grid_geom = grid_feature.geometry()
            grid_area = grid_geom.area()

            if grid_area == 0:
                continue

            building_area_total = self._calculate_building_area(
                grid_geom, building_index
            )
            building_percent = (building_area_total / grid_area) * 100

            if building_percent > 5:
                new_feature = QgsFeature(result_layer.fields())
                new_feature.setGeometry(grid_geom)
                new_feature.setAttributes(
                    grid_feature.attributes() + [building_percent]
                )
                features_to_add.append(new_feature)

            progress.setValue(idx + 1)

        provider.addFeatures(features_to_add)
        progress.close()

        LayerManager.get_project().addMapLayer(result_layer)
        LayerManager.move_layer_to_top(result_layer)
        iface.mapCanvas().refresh()

    def _calculate_building_area(self, grid_geom, building_index):
        intersecting_ids = building_index.intersects(grid_geom.boundingBox())
        building_area_total = 0

        for building_id in intersecting_ids:
            building_feature = self.buildings_layer.getFeature(building_id)
            building_geom = building_feature.geometry()

            if grid_geom.intersects(building_geom):
                intersection = grid_geom.intersection(building_geom)
                if not intersection.isEmpty():
                    building_area_total += intersection.area()

        return building_area_total


def main():
    analyzer = GridAnalyzer()
    analyzer.reproject_layers_to_2180()

    print("Reordering layers...")
    LayerManager.reorder_layers(Config.PREFERRED_LAYER_ORDER, Config.GRID_LAYER_NAME)

    print("Applying styles...")
    StyleManager.apply_styles(Config.STYLES)

    print("Starting grid analysis...")
    analyzer.analyze_grid()

    print("Process completed successfully")


main()
