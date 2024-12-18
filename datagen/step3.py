import os
import shutil
from pathlib import Path
from qgis.core import QgsProject
from qgis.utils import iface

ROOT_DIR = Path.home()/Path("Pulpit/drony/")
ORTO_DIR = ROOT_DIR/Path("qgisdownload")

def clear_project_and_downloads():
    """Usuwa wszystkie warstwy z projektu i czyści folder"""
    
    project = QgsProject.instance()
    project.removeAllMapLayers()
    print("Usunięto wszystkie warstwy z projektu")
    
    try:
        if ORTO_DIR.exists():
            shutil.rmtree(str(ORTO_DIR))
            ORTO_DIR.mkdir(parents=True)
            print(f"Wyczyszczono folder: {ORTO_DIR}")
        else:
            print(f"Folder {ORTO_DIR} nie istnieje")
            
    except Exception as e:
        print(f"Wystąpił błąd podczas czyszczenia folderu: {str(e)}")
    
    iface.mapCanvas().refresh()
    print("Zakończono czyszczenie projektu i folderu")


clear_project_and_downloads()