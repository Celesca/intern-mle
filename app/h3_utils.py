import h3

# H3 resolution to average edge length (in meters)
# Resolution 7: ~1.22 km edge, good for 5km radius searches
# Resolution 8: ~461 m edge, good for 1-2km radius searches
# Resolution 9: ~174 m edge, good for <500m radius searches
H3_RESOLUTION_MAP = {
    500: 9,    # <500m: use resolution 9
    2000: 8,   # 500m-2km: use resolution 8
    5000: 7,   # 2km-5km: use resolution 7
    10000: 6,  # 5km-10km: use resolution 6
    50000: 5,  # 10km-50km: use resolution 5
}


def get_resolution_for_distance(max_dist_meters: float) -> int:
    for threshold, resolution in sorted(H3_RESOLUTION_MAP.items()):
        if max_dist_meters <= threshold:
            return resolution
    return 4  # Very large distances


def lat_lng_to_h3(lat: float, lng: float, resolution: int) -> str:
    return h3.geo_to_h3(lat, lng, resolution)


def get_h3_cells_in_radius(lat: float, lng: float, radius_meters: float, resolution: int) -> set:
    center_cell = h3.geo_to_h3(lat, lng, resolution)
    
    edge_length_km = h3.edge_length(resolution, unit='km')
    edge_length_m = edge_length_km * 1000
    
    k = int(radius_meters / edge_length_m) + 2
    
    return h3.k_ring(center_cell, k)


def filter_by_h3_proximity(
    user_lat: float,
    user_lng: float,
    restaurant_lats: list,
    restaurant_lngs: list,
    restaurant_ids: list,
    max_dist_meters: float
) -> tuple:
    resolution = get_resolution_for_distance(max_dist_meters)
    
    nearby_cells = get_h3_cells_in_radius(user_lat, user_lng, max_dist_meters, resolution)
    
    filtered_indices = []
    for i, (lat, lng) in enumerate(zip(restaurant_lats, restaurant_lngs)):
        cell = h3.geo_to_h3(lat, lng, resolution)
        if cell in nearby_cells:
            filtered_indices.append(i)
    
    return filtered_indices
