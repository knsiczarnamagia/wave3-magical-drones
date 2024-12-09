import os
import shutil
from pathlib import Path
from qgis.core import QgsProject
from qgis.utils import iface

ROOT_DIR = Path.home()/Path("Pulpit/drony/")
ORTO_DIR = ROOT_DIR/Path("qgisdownload")

def clear_project_and_downloads():
    """Usuwa wszystkie warstwy z projektu i czyści folder"""
    
    # Usuwanie wszystkich warstw z projektu
    project = QgsProject.instance()
    project.removeAllMapLayers()
    print("Usunięto wszystkie warstwy z projektu")
    
    try:
        # Sprawdzenie czy folder istnieje
        if ORTO_DIR.exists():
            # Usunięcie zawartości folderu
            shutil.rmtree(str(ORTO_DIR))
            # Utworzenie pustego folderu
            ORTO_DIR.mkdir(parents=True)
            print(f"Wyczyszczono folder: {ORTO_DIR}")
        else:
            print(f"Folder {ORTO_DIR} nie istnieje")
            
    except Exception as e:
        print(f"Wystąpił błąd podczas czyszczenia folderu: {str(e)}")
    
    # Odświeżenie interfejsu
    iface.mapCanvas().refresh()
    print("Zakończono czyszczenie projektu i folderu")


clear_project_and_downloads()