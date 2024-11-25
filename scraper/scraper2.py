# Zoom | Max Tiles | Approximate Range
# ---------------------------------------------
#   13 |  2500x2500  | 0-2500
#   12 |  1250x1250  | 0-1250
#   11 |   625x625   | 0-625
#   10 |   312x312   | 0-312
#    9 |   156x156   | 0-156
#    8 |    78x78    | 0-78
#    7 |    39x39    | 0-39
#    6 |    19x19    | 0-19
#    5 |     9x9     | 0-9
#    4 |     4x4     | 0-4
#    3 |     2x2     | 0-2
#    2 |     1x1     | 0-1
#    1 |     0x0     | 0-0


import requests
import shutil, time, random
# from pathlib import Path # clear dirs before gathering

# shutil.rmtree('maps/orto/')
# Path('maps/orto').mkdir(parents=True, exist_ok=True)
# shutil.rmtree('maps/topo/')
# Path('maps/topo').mkdir(parents=True, exist_ok=True)

max_zoom = 13
for i in range(20):
    zoom = random.randrange(5, max_zoom) # min: 1 max: 14
    min_tile_for_zoom, max_tile_for_zoom = \
        2400 // (2 ** (13 - (zoom-1))), 2400 // (2 ** (13 - zoom))
    x, y = random.randrange(min_tile_for_zoom), \
        random.randrange(min_tile_for_zoom, max_tile_for_zoom)

    orto_url = f'https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/REST/StandardResolution/tile/{zoom}/{x}/{y}'
    orto_headers = {
        'Referer': 'http://mapy.geoportal.gov.pl/',
        'DNT': '1',
    }
    orto_response = requests.get(orto_url, headers=orto_headers) # I RECOMMEND USING ROTATING PROXY (e.g. webshare.io free proxies)

    topo_url = f'https://mapy.geoportal.gov.pl/gprest/services/G2_MOBILE_500/MapServer/tile/{zoom}/{x}/{y}'
    topo_headers = {
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'DNT': '1',
        'Referer': 'http://mapy.geoportal.gov.pl/',
        'Sec-Fetch-Dest': 'image',
        'Sec-Fetch-Mode': 'no-cors',
    }
    topo_response = requests.get(topo_url, headers=topo_headers) # I RECOMMEND USING ROTATING PROXY (e.g. webshare.io free proxies)

    # print(f'{orto_response.status_code}, x={x}, y={y}, z={zoom}')
    # if int(topo_response.headers['Content-Length']) < 1000:
    #     print(len(topo_response.content))

    if orto_response.status_code < 300 and topo_response.status_code < 300 and int(topo_response.headers['Content-Length']) > 1000:
            with open(f'maps/orto/{x}_{y}_{zoom}.png', 'wb') as file:
                file.write(orto_response.content)

            with open(f'maps/topo/{x}_{y}_{zoom}.png', 'wb') as file:
                file.write(topo_response.content)

    time.sleep(random.random()/2+0.2) # 0.2-0.7s
