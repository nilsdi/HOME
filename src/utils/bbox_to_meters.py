from pyproj import Transformer


def convert_bbox_to_meters(bbox, source_crs='EPSG:4326',
                           target_crs='EPSG:25833'):
    transformer = Transformer.from_crs(source_crs, target_crs)
    left, bottom = transformer.transform(bbox[0], bbox[1])
    right, top = transformer.transform(bbox[2], bbox[3])
    return [left, bottom, right, top]
