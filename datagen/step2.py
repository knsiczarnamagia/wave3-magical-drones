from utils import Config, LayerManager, StyleManager
from pathlib import Path
import uuid
import os
from qgis.core import (
    QgsMapLayer,
    QgsMapSettings,
    QgsMapRendererCustomPainterJob,
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QProgressDialog
from PyQt5.QtGui import QImage, QPainter
from qgis.utils import iface


class ViewRenderer:
    def __init__(self):
        self._create_dirs()
        self.grid_layer = LayerManager.get_project().mapLayersByName(
            Config.GRID_ANALYSIS_LAYER_NAME
        )[0]
        self.layers = LayerManager.get_project().mapLayers().values()

    def _create_dirs(self):
        place_id = str(uuid.uuid4())[:8]
        self.raster_out_dir = Config.ROOT_DIR / Path("sat", place_id)
        self.vector_out_dir = Config.ROOT_DIR / Path("map", place_id)
        self.raster_out_dir.mkdir(parents=True, exist_ok=True)
        self.vector_out_dir.mkdir(parents=True, exist_ok=True)

    def render_views(self, image_width: int = 500, image_height: int = 500):
        progress_dialog = QProgressDialog(
            "Rendering views...",
            "Cancel",
            0,
            self.grid_layer.featureCount(),
            iface.mainWindow(),
        )
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.setMinimumDuration(0)
        progress_dialog.setWindowTitle("Processing")
        progress_dialog.show()

        for idx, feature in enumerate(self.grid_layer.getFeatures()):
            if progress_dialog.wasCanceled():
                break

            extent = feature.geometry().boundingBox()
            feature_id = feature.id()

            vector_layers = []
            raster_layers = []

            for layer in self.layers:
                if extent.intersects(layer.extent()):
                    if layer.type() == QgsMapLayer.VectorLayer:
                        vector_layers.append(layer)
                    elif layer.type() == QgsMapLayer.RasterLayer:
                        raster_layers.append(layer)

            layer_configs = [
                (vector_layers, self.vector_out_dir, "vectors"),
                (raster_layers, self.raster_out_dir, "rasters"),
            ]

            for layers, out_dir, layer_type in layer_configs:
                if layers:
                    self._render_layer_group(
                        layers,
                        extent,
                        image_width,
                        image_height,
                        out_dir,
                        feature_id,
                        layer_type,
                    )

            progress_dialog.setValue(idx + 1)
            iface.mainWindow().repaint()

        progress_dialog.close()
        print("Rendering completed")

    def _render_layer_group(
        self,
        layers: list,
        extent,
        width: int,
        height: int,
        output_folder: Path,
        id,
        layer_type: str,
    ):
        map_settings = QgsMapSettings()
        map_settings.setLayers(layers)
        map_settings.setExtent(extent)
        map_settings.setOutputSize(QSize(width, height))
        map_settings.setBackgroundColor(Qt.white)
        map_settings.setOutputDpi(96)

        image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.white)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)

        render_job = QgsMapRendererCustomPainterJob(map_settings, painter)
        render_job.start()
        render_job.waitForFinished()
        painter.end()

        output_path = os.path.join(str(output_folder), f"cell_{id}.png")
        success = image.save(output_path)

        if success:
            print(
                f"Successfully saved {layer_type} image for ID: {id} at {output_path}"
            )
        else:
            print(f"Failed to save {layer_type} image for ID: {id}")


def main():
    LayerManager.clear_all_selections()
    StyleManager.set_grid_transparent(Config.GRID_ANALYSIS_LAYER_NAME)
    StyleManager.set_grid_transparent(
        Config.GRID_LAYER_NAME
    )  # it could be also removed completely
    LayerManager.reorder_layers(
        Config.PREFERRED_LAYER_ORDER, Config.GRID_ANALYSIS_LAYER_NAME
    )

    renderer = ViewRenderer()
    renderer.render_views()


main()
