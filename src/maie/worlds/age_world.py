import math
import random
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Iterable

import numpy as np
import pygame
from scipy.spatial import Delaunay

from maie.camera import RenderContext, DrawLayer, draw_tile
from maie.common import Vec2, Color, GVec2, clamp
from maie.nearest_neighbors import NearestNeighbors
from maie.perlin import perlin2d, perlin2d_fbm
from maie.poisson import poisson_disc_2d
from maie.voronoi import voronoi_partition_tiles
from maie.tilesets.tile_age import TileType, get_tileset, TileColors

@dataclass
class AgeWorldConfig:
    width: int = 1024
    height: int = 768
    tile_size: float = 8.
    seed: int = 42
    poisson_n_points: int = 150
    poisson_radius: float = 80
    
    # Terrain
    land_threshold: float = 0.1  # Higher = more water (less land)
    k_nearest: int = 6
    sigma: float = 800.0  # Gaussian smoothing radius
    
    # Resources
    gold_scarcity: float = 0.98  # Rarer gold clusters
    stone_scarcity: float = 0.85  # Rarer stone clusters
    wood_threshold: float = 0.45  # Lower threshold = more wood
    wood_cluster_chance: float = 0.7  # Higher chance = more wood spreading
    n_starting_points: int = 4  # Number of starting points (cities)
    
    n_cities: int = 0  # Don't auto-generate cities in terrain
    city_size: int = 80 # Tiles
    city_min_dist: float = 150  # Minimum distance between cities
    river_width: int = 5  # River width in tiles (bigger, more visible)
    n_rivers: int = 4  # Number of rivers to generate

class AgeWorld:
    def __init__(self, cfg: AgeWorldConfig = AgeWorldConfig()):
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        self.ts = cfg.tile_size
        self.tileset = get_tileset(self.ts)
        
        self._regenerate()

    def get_layers(self, which: Iterable[int]) -> list[DrawLayer]:
        # Linear pipeline visualization (1-7 follow generation steps)
        # 1 = Poisson Seeds, 2 = Graph, 3 = Voronoi, 4 = Rivers, 5 = Resources, 6 = Final, 7 = Cities
        layer_map = {
            1: DrawLayer(z=10, label="step1_poisson_seeds", draw=self._draw_poisson_seeds),
            2: DrawLayer(z=20, label="step2_delaunay_graph", draw=self._draw_graph),
            3: DrawLayer(z=30, label="step3_voronoi_regions", draw=self._draw_voronoi_regions),
            4: DrawLayer(z=40, label="step4_rivers", draw=self._draw_rivers),
            5: DrawLayer(z=50, label="step5_resources", draw=self._draw_resources_graph),
            6: DrawLayer(z=60, label="step6_final_raster", draw=self._draw_terrain),
            7: DrawLayer(z=70, label="step7_cities", draw=self._draw_cities),
        }
        
        # If no layers specified or empty, show default final view (step 6)
        if not which:
            return [layer_map[6]]
        
        # Return requested layers in order
        result = []
        for idx in sorted(which):
            if idx in layer_map:
                result.append(layer_map[idx])
        # If no valid layers found, default to final raster
        return result if result else [layer_map[6]]

    def debug_layers(self) -> list[DrawLayer]:
        return [
            DrawLayer(z=50, label="graph", draw=self._draw_graph),
            DrawLayer(z=51, label="centers", draw=self._draw_centers),
        ]

    def _regenerate(self):
        # 1. Poisson Seeds
        self._generate_points()
        
        # 2. Delaunay Graph
        self._build_graph()
        
        # 3. Regional Values (Terrain)
        self._generate_regional_terrain()

        # 3b. Rivers
        self._generate_rivers()
        
        # 4. Initial Rasterization (to get accurate region elevations)
        self._do_initial_rasterization()
        
        # 5. Resources (placed after rasterization so we have accurate elevations)
        self._distribute_resources()

        # 6. Generate starting points (cities) with resource clusters nearby
        # (must happen before rasterization so gold/stone become tiles)
        self._generate_starting_points_with_resources()
        
        # 7. Final Rasterization with resources
        self._rasterize_with_resources()

        # 8. Finalize city tiles (after terrain exists)
        self._finalize_city_tiles()

    def _generate_points(self):
        sites = poisson_disc_2d(
            (0, 0, self.cfg.width, self.cfg.height),
            radius=self.cfg.poisson_radius,
            n_points=self.cfg.poisson_n_points,
            seed=self.cfg.seed
        )
        self.sites = sites
        self.n_sites = len(sites)
        
        # Initialize NearestNeighbors for smoothing
        self.neighbors = NearestNeighbors(sites)

    def _build_graph(self):
        # Delaunay triangulation
        self.tri = Delaunay(self.sites)
        
        # Build adjacency list
        self.adj = [set() for _ in range(self.n_sites)]
        for simplex in self.tri.simplices:
            for i in range(3):
                u, v = simplex[i], simplex[(i+1)%3]
                self.adj[u].add(v)
                self.adj[v].add(u)

    def _generate_regional_terrain(self):
        rng = random.Random(self.cfg.seed)
        perlin_offset = rng.uniform(-0.5, 0.5)
        
        # First, assign rough elevation/moisture to each region center
        self.site_elev = {}
        self.site_moisture = {}
        self.site_is_land_pref = {}
        
        for i, (x, y) in enumerate(self.sites):
            # Sample Perlin noise at multiple scales for natural-looking distribution
            val = 0.5 * (perlin2d(x / 256, y / 256, seed=self.cfg.seed) + 1)
            # More water (higher threshold = more water regions)
            is_land_pref = val > 0.40  # Rough land/sea preference (more water)
            
            if is_land_pref:
                # Land: higher base elevation
                base_elev = rng.uniform(0.35, 0.70)
            else:
                # Sea: lower base elevation
                base_elev = rng.uniform(-0.3, 0.10)
            
            self.site_elev[i] = base_elev
            self.site_is_land_pref[i] = is_land_pref
            
            # Moisture for land regions
            m = perlin2d(x / 512, y / 512, seed=self.cfg.seed + 100)
            self.site_moisture[i] = (m + 1) / 2.0
        
        # Now interpolate to tiles using KNN (like UUWorld)
        shape_tiles = int(math.floor(self.width / self.ts)), int(math.floor(self.height / self.ts))
        self.elevation_map = np.zeros(shape_tiles, dtype=float)
        self.moisture_map = np.zeros(shape_tiles, dtype=float)
        self.land_pref_map = np.zeros(shape_tiles, dtype=float)
        self.is_land = np.zeros(shape_tiles, dtype=bool)
        
        def b_to_f(val: bool) -> float:
            return 1.0 if val else -1.0
        
        for tile in np.ndindex(shape_tiles):
            tx, ty = tile
            point = (tx * self.ts + self.ts/2, ty * self.ts + self.ts/2)
            
            # Get k-nearest neighbors
            k = self.cfg.k_nearest
            nn = self.neighbors.nearest_neighbors(point, k)
            
            # Get owners for each neighbor
            owners = []
            for p in nn:
                # Find which site this point belongs to (approximate)
                best_owner = 0
                best_d2 = float('inf')
                for site_idx, site in enumerate(self.sites):
                    dx = p[0] - site[0]
                    dy = p[1] - site[1]
                    d2 = dx*dx + dy*dy
                    if d2 < best_d2:
                        best_d2 = d2
                        best_owner = site_idx
                owners.append(best_owner)
            
            # Compute weights (Gaussian)
            weights = []
            for p in nn:
                tile_point = np.array([point[0] / self.ts, point[1] / self.ts])
                p_point = np.array([p[0] / self.ts, p[1] / self.ts])
                diff = np.linalg.norm(tile_point - p_point)
                w = np.exp(-diff**2 / (2 * (self.cfg.sigma / self.ts)**2))
                weights.append(w)
            
            if sum(weights) > 0:
                weights = np.array([w / sum(weights) for w in weights])
            else:
                weights = np.array([1.0 / len(weights)] * len(weights))
            
            # Interpolate elevation
            elevs = np.array([self.site_elev[o] for o in owners])
            self.elevation_map[tile] = np.dot(weights, elevs)
            
            # Interpolate moisture
            mois = np.array([self.site_moisture[o] for o in owners])
            self.moisture_map[tile] = np.dot(weights, mois)
            
            # Interpolate land preference
            land_prefs = np.array([b_to_f(self.site_is_land_pref[o]) for o in owners])
            land_pref = np.dot(weights, land_prefs)
            self.land_pref_map[tile] = land_pref
            
            # Add Perlin detail noise
            x, y = point
            x += perlin_offset
            y += perlin_offset
            detail = 0.3 * perlin2d_fbm(x / 128, y / 128, octaves=3, lacunarity=2, gain=0.5, seed=self.cfg.seed + 200)
            self.elevation_map[tile] += detail
            
        # Smooth elevation/moisture for more realistic landmasses
        self.elevation_map = self._smooth_map(self.elevation_map, passes=2)
        self.moisture_map = self._smooth_map(self.moisture_map, passes=1)

        # Determine land/water after smoothing
        for tile in np.ndindex(shape_tiles):
            land_pref = self.land_pref_map[tile]
            self.is_land[tile] = (land_pref > self.cfg.land_threshold) and (self.elevation_map[tile] > 0.25)
        
        # Now assign region-level values for resource placement (average of tiles in region)
        # But we also need per-region values for graph traversal, so average tile values per owner
        self.region_elev = np.zeros(self.n_sites)
        self.region_moisture = np.zeros(self.n_sites)
        self.region_is_land = np.zeros(self.n_sites, dtype=bool)
        
        # We'll compute these during rasterization when we have owner_map
        # For now, use site values as fallback
        for i in range(self.n_sites):
            self.region_elev[i] = self.site_elev[i]
            self.region_moisture[i] = self.site_moisture[i]
            self.region_is_land[i] = self.site_is_land_pref[i]

    def _smooth_map(self, arr: np.ndarray, passes: int = 1) -> np.ndarray:
        """Simple box blur to smooth terrain fields."""
        smoothed = arr
        for _ in range(passes):
            padded = np.pad(smoothed, 1, mode="edge")
            out = np.empty_like(smoothed)
            for x in range(out.shape[0]):
                for y in range(out.shape[1]):
                    window = padded[x:x+3, y:y+3]
                    out[x, y] = float(np.mean(window))
            smoothed = out
        return smoothed

    def _generate_rivers(self):
        self.river_edges = set() # (u, v) where u < v
        
        # Springs: High elev, High moisture - more lenient criteria for more rivers
        # Rivers start on land but need to reach sea
        springs_candidates = [
            i for i in range(self.n_sites) 
            if self.region_is_land[i] and self.region_elev[i] > 0.4 and self.region_moisture[i] > 0.3
        ]
        
        # Sort by elevation + moisture (best springs first)
        springs_candidates.sort(key=lambda i: self.region_elev[i] + self.region_moisture[i], reverse=True)
        
        # Select top N springs, ensuring some distance between them
        springs = []
        min_spring_dist = 150  # Minimum distance between springs (pixels)
        
        for candidate in springs_candidates[:self.cfg.n_rivers * 2]:  # Consider more candidates
            if len(springs) >= self.cfg.n_rivers:
                break
            
            pos = self.sites[candidate]
            too_close = False
            
            for existing_spring_idx in springs:
                existing_pos = self.sites[existing_spring_idx]
                dist = math.sqrt((pos[0] - existing_pos[0])**2 + (pos[1] - existing_pos[1])**2)
                if dist < min_spring_dist:
                    too_close = True
                    break
            
            if not too_close:
                springs.append(candidate)
        
        # Generate river paths from each spring
        for start_node in springs:
            curr = start_node
            max_length = 30  # Limit river length to avoid very long rivers
            length = 0
            
            while length < max_length:
                # Find lowest neighbor (prefer non-land to reach sea)
                best_n = -1
                min_elev = self.region_elev[curr]
                sea_found = False
                
                # First, check if any neighbor is sea (water)
                for n in self.adj[curr]:
                    if not self.region_is_land[n]:
                        best_n = n
                        sea_found = True
                        break
                
                # If no sea neighbor, find lowest elevation neighbor
                if not sea_found:
                    for n in self.adj[curr]:
                        if self.region_elev[n] < min_elev:
                            min_elev = self.region_elev[n]
                            best_n = n
                
                if best_n != -1:
                    # Flow downhill
                    u, v = tuple(sorted((curr, best_n)))
                    if (u, v) not in self.river_edges:  # Don't duplicate edges
                        self.river_edges.add((u, v))
                    
                    if not self.region_is_land[best_n]:
                        break # Reached sea
                    curr = best_n
                    length += 1
                else:
                    break # Local minimum (lake?)

    def _do_initial_rasterization(self):
        """Do initial rasterization to get accurate region elevations for resource placement"""
        # Create grid map
        self.owner_map, (self.w_tiles, self.h_tiles), _ = voronoi_partition_tiles(
            (0, 0, self.width, self.height),
            self.sites,
            self.ts
        )
        
        # Update region-level values based on actual tile averages
        region_elev_sum = np.zeros(self.n_sites)
        region_moisture_sum = np.zeros(self.n_sites)
        region_tile_counts = np.zeros(self.n_sites, dtype=int)
        
        for tile_coord, owner_idx in self.owner_map.items():
            if tile_coord[0] < self.w_tiles and tile_coord[1] < self.h_tiles:
                region_elev_sum[owner_idx] += self.elevation_map[tile_coord]
                region_moisture_sum[owner_idx] += self.moisture_map[tile_coord]
                region_tile_counts[owner_idx] += 1
        
        for i in range(self.n_sites):
            if region_tile_counts[i] > 0:
                self.region_elev[i] = region_elev_sum[i] / region_tile_counts[i]
                self.region_moisture[i] = region_moisture_sum[i] / region_tile_counts[i]
                # Region is land if majority of its tiles are land
                land_count = sum(1 for (tx, ty), owner in self.owner_map.items() 
                               if owner == i and tx < self.w_tiles and ty < self.h_tiles and self.is_land[(tx, ty)])
                self.region_is_land[i] = land_count > region_tile_counts[i] * 0.5

    def _distribute_resources(self):
        # 0: None, 1: Wood, 2: Stone, 3: Gold
        self.region_resource = np.zeros(self.n_sites, dtype=int)
        
        indices = list(range(self.n_sites))
        
        # Wood: More wood with tighter spacing (stronger clustering)
        wood_seed_candidates = [i for i in indices if self.region_is_land[i] and self.region_moisture[i] > self.cfg.wood_threshold and self.region_resource[i] == 0]
        if wood_seed_candidates:
            random.shuffle(wood_seed_candidates)
            # More seeds for more forest presence
            n_seeds = max(4, int(len(wood_seed_candidates) * 0.35))
            wood_seeds = wood_seed_candidates[:n_seeds]
            
            # More aggressive spreading
            visited_wood = set(wood_seeds)
            for seed in wood_seeds:
                self.region_resource[seed] = 1  # WOOD
                # Spread to neighbors (more aggressively)
                queue = [(seed, 0)]
                while queue:
                    curr, depth = queue.pop(0)
                    if depth >= 2:  # Spread up to 2 hops
                        continue
                    
                    for neighbor in self.adj[curr]:
                        if neighbor in visited_wood:
                            continue
                        if not self.region_is_land[neighbor]:
                            continue
                        if self.region_resource[neighbor] != 0:
                            continue
                        
                    # Slightly looser moisture requirement
                    if self.region_moisture[neighbor] > self.cfg.wood_threshold * 0.85:
                            if random.random() < self.cfg.wood_cluster_chance:
                                self.region_resource[neighbor] = 1  # WOOD
                                visited_wood.add(neighbor)
                                queue.append((neighbor, depth + 1))

    def _rasterize_with_resources(self):
        # Create tile type map with resources
        self.tile_type_map = {} # (x, y) -> TileType
        
        rng = random.Random(self.cfg.seed)
        
        for tile_coord, owner_idx in self.owner_map.items():
            tx, ty = tile_coord
            if not (tx < self.w_tiles and ty < self.h_tiles):
                continue
                
            # Use tile-level land/water decision
            if not self.is_land[tile_coord]:
                self.tile_type_map[tile_coord] = TileType.WATER
                continue
                
            res = self.region_resource[owner_idx]
            base = TileType.GRASS
            
            # Moisture bias for base (use tile-level moisture)
            if self.moisture_map[tile_coord] < 0.3:
                base = TileType.SAND
            
            # Elevation affects base - show mountains clearly
            if self.elevation_map[tile_coord] > 0.70:
                base = TileType.MOUNTAIN
            elif self.elevation_map[tile_coord] > 0.60:
                # Hills - mix of grass and mountain
                if rng.random() < 0.4:
                    base = TileType.MOUNTAIN
            
            if res == 3: # Gold
                # Smaller gold clusters (still visible)
                if rng.random() < 0.15:
                    self.tile_type_map[tile_coord] = TileType.GOLD
                elif self.elevation_map[tile_coord] > 0.65:
                    self.tile_type_map[tile_coord] = TileType.MOUNTAIN
                else:
                    self.tile_type_map[tile_coord] = base
            elif res == 2: # Stone
                # Stone should show as mountains - higher density
                if self.elevation_map[tile_coord] > 0.55:
                    self.tile_type_map[tile_coord] = TileType.MOUNTAIN
                elif rng.random() < 0.40:  # Higher chance for mountains in stone regions
                    self.tile_type_map[tile_coord] = TileType.MOUNTAIN
                else:
                    self.tile_type_map[tile_coord] = base
            elif res == 1: # Wood
                # More wood with tighter spacing for natural borders
                # Check if neighbors also have wood for clustering
                has_wood_neighbor = False
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    ntx, nty = tx + dx, ty + dy
                    if (ntx, nty) in self.owner_map:
                        no = self.owner_map[(ntx, nty)]
                        if self.region_resource[no] == 1:
                            has_wood_neighbor = True
                            break
                
                # Higher density - more wood, less spacing between trees
                density = 0.55 if has_wood_neighbor else 0.4
                if rng.random() < density:
                    self.tile_type_map[tile_coord] = TileType.FOREST
                else:
                    self.tile_type_map[tile_coord] = TileType.GRASS
            else:
                self.tile_type_map[tile_coord] = base

        # Rasterize Rivers (overwrite terrain) - WIDER RIVERS
        river_width = self.cfg.river_width
        for u, v in self.river_edges:
            p1 = self.sites[u]
            p2 = self.sites[v]
            
            # Line rasterization with width
            x1, y1 = p1
            x2, y2 = p2
            dist = math.hypot(x2-x1, y2-y1)
            steps = max(1, int(dist / (self.ts * 0.5)))
            
            # Calculate perpendicular direction for width
            dx = (x2 - x1) / dist if dist > 0 else 0
            dy = (y2 - y1) / dist if dist > 0 else 0
            perp_x = -dy
            perp_y = dx
            
            for i in range(steps + 1):
                t = i / steps if steps > 0 else 0
                lx = x1 + (x2 - x1) * t
                ly = y1 + (y2 - y1) * t
                
                # Add width perpendicular to the river direction
                for w in range(-river_width, river_width + 1):
                    width_offset = w * self.ts * 0.5
                    rx = lx + perp_x * width_offset
                    ry = ly + perp_y * width_offset
                    
                    tx, ty = int(rx/self.ts), int(ry/self.ts)
                    if 0 <= tx < self.w_tiles and 0 <= ty < self.h_tiles:
                        self.tile_type_map[(tx, ty)] = TileType.WATER

    def _place_cities(self):
        # 1. Score regions
        scores = {}
        for i in range(self.n_sites):
            if not self.region_is_land[i]:
                scores[i] = -1
                continue
                
            # Prefer flat land near resources
            score = 0
            if self.region_elev[i] < 0.5: score += 10 # Prefer flat
            
            # Scan neighbors for resources (Graph logic!)
            # Radius 2 hops
            visited = {i}
            queue = [(i, 0)]
            
            resources_nearby = {1:0, 2:0, 3:0} # Wood, Stone, Gold
            
            idx = 0
            while idx < len(queue):
                curr, dist = queue[idx]
                idx += 1
                
                # Count resource
                r = self.region_resource[curr]
                if r > 0:
                    resources_nearby[r] += 1
                
                if dist < 2:
                    for neighbor in self.adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist+1))
                            
            # "Gold is most important"
            score += resources_nearby[3] * 50
            # "Stone and wood is the same importance"
            score += resources_nearby[2] * 10
            score += resources_nearby[1] * 10
            
            scores[i] = score
            
        # Select Top N cities
        sorted_sites = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        self.cities = []
        min_dist = self.cfg.city_min_dist
        
        for site_idx in sorted_sites:
            if scores[site_idx] <= 0: 
                # If we still need cities but scores are low, be less strict
                if len(self.cities) < self.cfg.n_cities // 2:
                    # Accept lower scores for first half of cities
                    pass
                else:
                    break
            if len(self.cities) >= self.cfg.n_cities: break
            
            pos = self.sites[site_idx]
            
            # Check distance to existing cities
            too_close = False
            for c in self.cities:
                d = math.sqrt((c['pos'][0]-pos[0])**2 + (c['pos'][1]-pos[1])**2)
                if d < min_dist:
                    too_close = True
                    break
            
            if not too_close:
                city_tiles = self._grow_city(pos)
                # Only add if city actually grew (has tiles)
                if len(city_tiles) > 10:  # Minimum viable city size
                    self.cities.append({
                        'idx': site_idx,
                        'pos': pos,
                        'tiles': city_tiles
                    })

    def _grow_city(self, center_pos):
        # Organic city growth (BFS on tiles)
        cx, cy = int(center_pos[0]/self.ts), int(center_pos[1]/self.ts)
        start = (cx, cy)
        
        city_tiles = set()
        queue = [start]
        visited = {start}
        
        target_size = self.cfg.city_size
        
        while len(city_tiles) < target_size and queue:
            # Pick random from queue to make it less square? 
            # Or shuffle neighbors?
            idx = random.randint(0, len(queue)-1)
            curr = queue.pop(idx)
            
            city_tiles.add(curr)
            
            # Neighbors
            nbs = [
                (curr[0]+1, curr[1]), (curr[0]-1, curr[1]),
                (curr[0], curr[1]+1), (curr[0], curr[1]-1)
            ]
            random.shuffle(nbs)
            
            for nb in nbs:
                if nb in visited: continue
                # Bounds check
                if not (0 <= nb[0] < self.w_tiles and 0 <= nb[1] < self.h_tiles): continue
                
                # Check terrain - don't build on water or mountains
                t = self.tile_type_map.get(nb, TileType.WATER)
                if t == TileType.WATER or t == TileType.MOUNTAIN or t == TileType.GOLD:
                    continue
                
                visited.add(nb)
                queue.append(nb)
                
        return city_tiles

    # --- Starting Points with Resource Clusters ---
    
    def _generate_starting_points_with_resources(self):
        """Generate starting points (cities) with small gold/stone clusters nearby, and one big cluster"""
        rng = random.Random(self.cfg.seed + 3000)
        self.cities = []
        
        # Find suitable starting points: land, flat, no resource, access to wood or water,
        # and close to gold/stone in graph.
        starting_candidates = []
        for i in range(self.n_sites):
            if not self.region_is_land[i]:
                continue
            if not (0.2 < self.region_elev[i] < 0.6):
                continue
            if self.region_resource[i] != 0:
                continue

            gold_count = 0
            stone_count = 0
            wood_count = 0
            water_access = False

            visited = {i}
            queue = [(i, 0)]
            while queue:
                curr, dist = queue.pop(0)
                if dist > 2:
                    continue
                r = self.region_resource[curr]
                if r == 3:
                    gold_count += 1
                elif r == 2:
                    stone_count += 1
                elif r == 1:
                    wood_count += 1
                if not self.region_is_land[curr]:
                    water_access = True
                if dist < 2:
                    for neighbor in self.adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))

            if wood_count == 0 and not water_access:
                continue

            score = (gold_count * 50) + (stone_count * 25) + (wood_count * 10)
            if water_access:
                score += 30
            score -= abs(self.region_elev[i] - 0.35) * 10
            starting_candidates.append((i, score))
        
        # Select starting points with minimum distance
        starting_points = []
        min_start_dist = max(self.cfg.city_min_dist, 220)
        
        starting_candidates.sort(key=lambda x: x[1], reverse=True)
        for candidate, _score in starting_candidates:
            if len(starting_points) >= self.cfg.n_starting_points:
                break
            
            pos = self.sites[candidate]
            too_close = False
            
            for existing_idx in starting_points:
                existing_pos = self.sites[existing_idx]
                dist = math.sqrt((pos[0] - existing_pos[0])**2 + (pos[1] - existing_pos[1])**2)
                if dist < min_start_dist:
                    too_close = True
                    break
            
            if not too_close:
                starting_points.append(candidate)
                # Create city marker at starting point
                self.cities.append({
                    'pos': self.sites[candidate],
                    'region_idx': candidate,
                    'tiles': set(),  # Cities are just markers (starting points)
                    'idx': len(self.cities),
                    'is_main': False
                })

                # If city is near gold, gold mine is considered taken
                for n in [candidate] + list(self.adj[candidate]):
                    if self.region_resource[n] == 3:
                        self.region_resource[n] = 0
        
        # Place small gold/stone clusters near each starting point
        cluster_radius_hops = 3  # 3 hops in graph (wider search)
        for start_idx in starting_points:
            visited = {start_idx}
            queue = [(start_idx, 0)]
            gold_placed = 0
            stone_placed = 0
            # Gold is not guaranteed near cities (often missing)
            gold_target = 1 if rng.random() < 0.5 else 0
            stone_target = 1  # Small stone cluster near start
            
            # Find nearby regions for small clusters
            gold_candidates = []
            stone_candidates = []
            while queue:
                curr, dist = queue.pop(0)
                if dist > cluster_radius_hops:
                    continue
                
                # Look for good spots for gold/stone near starting point
                if dist > 0 and self.region_resource[curr] == 0 and self.region_is_land[curr]:
                    # Lower thresholds to find more candidates
                    if self.region_elev[curr] > 0.50:  # Lowered from 0.55
                        gold_candidates.append((curr, self.region_elev[curr]))
                    if self.region_elev[curr] > 0.40:  # Lowered from 0.45
                        stone_candidates.append((curr, self.region_elev[curr]))
                
                if dist < cluster_radius_hops:
                    for neighbor in self.adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
            
            # Sort by elevation (best first) and place
            gold_candidates.sort(key=lambda x: x[1], reverse=True)
            stone_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Place gold
            for candidate_idx, _ in gold_candidates[:gold_target]:
                if gold_placed < gold_target:
                    self.region_resource[candidate_idx] = 3  # GOLD
                    gold_placed += 1
            
            # Place stone (avoid already placed gold)
            for candidate_idx, _ in stone_candidates[:stone_target * 2]:  # Check more candidates
                if stone_placed < stone_target and self.region_resource[candidate_idx] == 0:
                    self.region_resource[candidate_idx] = 2  # STONE
                    stone_placed += 1

        # Add a few extra small gold clusters (single-region) away from starts
        extra_gold_targets = 2
        extra_gold_candidates = [
            i for i in range(self.n_sites)
            if self.region_is_land[i] and self.region_elev[i] > 0.55 and self.region_resource[i] == 0
        ]
        rng.shuffle(extra_gold_candidates)
        for candidate in extra_gold_candidates:
            if extra_gold_targets <= 0:
                break
            # Keep away from starts for fairness
            too_close = False
            for start_idx in starting_points:
                dist = math.hypot(self.sites[candidate][0] - self.sites[start_idx][0],
                                  self.sites[candidate][1] - self.sites[start_idx][1])
                if dist < 180:
                    too_close = True
                    break
            if too_close:
                continue
            self.region_resource[candidate] = 3  # GOLD
            extra_gold_targets -= 1
        
        # Always create ONE BIG cluster of gold somewhere on the map (away from starting points)
        big_cluster_candidates = [
            i for i in range(self.n_sites) 
            if self.region_is_land[i] and self.region_elev[i] > 0.45 and self.region_resource[i] == 0
        ]
        
        if big_cluster_candidates and starting_points:
            # Choose a location away from starting points (fair distribution)
            best_cluster_center = None
            max_min_dist = 0
            
            # Sort by elevation first
            big_cluster_candidates.sort(key=lambda i: self.region_elev[i], reverse=True)
            
            for candidate in big_cluster_candidates[:50]:  # Check top 50 candidates
                pos = self.sites[candidate]
                min_dist_to_start = float('inf')
                for start_idx in starting_points:
                    start_pos = self.sites[start_idx]
                    dist = math.sqrt((pos[0] - start_pos[0])**2 + (pos[1] - start_pos[1])**2)
                    min_dist_to_start = min(min_dist_to_start, dist)
                
                # Prefer locations away from starting points (fair)
                if min_dist_to_start > max_min_dist and min_dist_to_start > 150:
                    max_min_dist = min_dist_to_start
                    best_cluster_center = candidate
            
            # If no good candidate found, just use the highest elevation one
            if best_cluster_center is None and big_cluster_candidates:
                best_cluster_center = big_cluster_candidates[0]
            
            if best_cluster_center is not None:
                # Create big cluster: spread gold and stone from center
                visited_big = {best_cluster_center}
                queue_big = [(best_cluster_center, 0)]
                
                gold_candidates_big = []
                
                while queue_big:
                    curr, dist = queue_big.pop(0)
                    if dist > 4:  # Spread up to 4 hops
                        continue
                    
                    if self.region_resource[curr] == 0 and self.region_is_land[curr]:
                        if self.region_elev[curr] > 0.50:
                            gold_candidates_big.append((curr, self.region_elev[curr]))
                    
                    if dist < 4:
                        for neighbor in self.adj[curr]:
                            if neighbor not in visited_big:
                                visited_big.add(neighbor)
                                queue_big.append((neighbor, dist + 1))
                
                # Sort and place big cluster
                gold_candidates_big.sort(key=lambda x: x[1], reverse=True)
                target_gold = 4
                
                for candidate_idx, _ in gold_candidates_big[:target_gold]:
                    if self.region_resource[candidate_idx] == 0:
                        self.region_resource[candidate_idx] = 3  # GOLD

        # Final rule: if a city is near gold, that gold is considered taken
        for city in self.cities:
            cidx = city["region_idx"]
            for n in [cidx] + list(self.adj[cidx]):
                if self.region_resource[n] == 3:
                    self.region_resource[n] = 0

    def _finalize_city_tiles(self):
        """Grow city tiles after terrain exists and apply to tile map."""
        if not getattr(self, "tile_type_map", None):
            return
        if not self.cities:
            return
        for city in self.cities:
            if city.get("tiles"):
                continue
            city_tiles = self._grow_city(city["pos"])
            city["tiles"] = city_tiles
            for tile in city_tiles:
                self.tile_type_map[tile] = TileType.CITY

        # Connect cities with paths after tiles exist
        self._connect_cities_with_paths()

    def _city_resource_access(self, region_idx: int, hops: int = 2):
        """Count nearby resources and water access from a region."""
        gold = 0
        stone = 0
        wood = 0
        water_access = False
        visited = {region_idx}
        queue = [(region_idx, 0)]
        while queue:
            curr, dist = queue.pop(0)
            if dist > hops:
                continue
            r = self.region_resource[curr]
            if r == 3:
                gold += 1
            elif r == 2:
                stone += 1
            elif r == 1:
                wood += 1
            if not self.region_is_land[curr]:
                water_access = True
            if dist < hops:
                for n in self.adj[curr]:
                    if n not in visited:
                        visited.add(n)
                        queue.append((n, dist + 1))
        return {"gold": gold, "stone": stone, "wood": wood, "water": water_access}

    def _find_region_path(self, start: int, goal: int):
        """BFS on region graph restricted to land; returns list of region indices."""
        if start == goal:
            return [start]
        visited = {start}
        parent = {start: None}
        queue = [start]
        while queue:
            curr = queue.pop(0)
            for n in self.adj[curr]:
                if n in visited:
                    continue
                if not self.region_is_land[n]:
                    continue
                visited.add(n)
                parent[n] = curr
                if n == goal:
                    queue = []
                    break
                queue.append(n)
        if goal not in parent:
            return [start]
        # Reconstruct
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return path

    def _connect_cities_with_paths(self):
        """Connect main city to all others, and extra paths for missing gold."""
        if not self.cities:
            return
        # Determine main city (best access to resources)
        best_idx = 0
        best_score = -1
        for i, city in enumerate(self.cities):
            access = self._city_resource_access(city["region_idx"], hops=2)
            score = access["gold"] * 50 + access["stone"] * 25 + access["wood"] * 10
            if access["water"]:
                score += 15
            if score > best_score:
                best_score = score
                best_idx = i
        for i, city in enumerate(self.cities):
            city["is_main"] = (i == best_idx)

        main_region = self.cities[best_idx]["region_idx"]
        path_edges = set()

        # Always connect main to every other city
        for i, city in enumerate(self.cities):
            if i == best_idx:
                continue
            region_idx = city["region_idx"]
            path = self._find_region_path(main_region, region_idx)
            for a, b in zip(path, path[1:]):
                u, v = tuple(sorted((a, b)))
                path_edges.add((u, v))

        # Extra paths for cities without gold
        cities_with_gold = []
        for i, city in enumerate(self.cities):
            access = self._city_resource_access(city["region_idx"], hops=2)
            if access["gold"] > 0:
                cities_with_gold.append(i)

        for i, city in enumerate(self.cities):
            access = self._city_resource_access(city["region_idx"], hops=2)
            if access["gold"] > 0:
                continue
            # Connect to nearest city that has gold
            best_target = None
            best_path = None
            for j in cities_with_gold:
                target_region = self.cities[j]["region_idx"]
                path = self._find_region_path(city["region_idx"], target_region)
                if best_path is None or len(path) < len(best_path):
                    best_path = path
                    best_target = j
            if best_path and best_target is not None:
                for a, b in zip(best_path, best_path[1:]):
                    u, v = tuple(sorted((a, b)))
                    path_edges.add((u, v))

        self._rasterize_paths(path_edges)

    def _rasterize_paths(self, path_edges):
        """Rasterize region-edge paths into PATH tiles."""
        if not path_edges:
            return
        def is_walkable(tile):
            current = self.tile_type_map.get(tile, TileType.WATER)
            return current in (TileType.GRASS, TileType.SAND, TileType.FOREST, TileType.CITY)

        def carve_axis_path(start, end, axis_first="x"):
            x0, y0 = start
            x1, y1 = end
            path = []
            if axis_first == "x":
                step = 1 if x1 >= x0 else -1
                for x in range(x0, x1 + step, step):
                    path.append((x, y0))
                step = 1 if y1 >= y0 else -1
                for y in range(y0, y1 + step, step):
                    path.append((x1, y))
            else:
                step = 1 if y1 >= y0 else -1
                for y in range(y0, y1 + step, step):
                    path.append((x0, y))
                step = 1 if x1 >= x0 else -1
                for x in range(x0, x1 + step, step):
                    path.append((x, y1))
            return path

        for u, v in path_edges:
            p1 = self.sites[u]
            p2 = self.sites[v]
            start = (int(p1[0] / self.ts), int(p1[1] / self.ts))
            end = (int(p2[0] / self.ts), int(p2[1] / self.ts))

            # Try axis-aligned paths only (no diagonals), avoid water/rivers
            candidate_paths = [
                carve_axis_path(start, end, axis_first="x"),
                carve_axis_path(start, end, axis_first="y"),
            ]

            chosen = None
            for cand in candidate_paths:
                blocked = False
                for tile in cand:
                    tx, ty = tile
                    if not (0 <= tx < self.w_tiles and 0 <= ty < self.h_tiles):
                        blocked = True
                        break
                    if not is_walkable(tile):
                        blocked = True
                        break
                if not blocked:
                    chosen = cand
                    break

            if chosen is None:
                # If both axis orders hit water, skip this edge
                continue

            for tx, ty in chosen:
                current = self.tile_type_map.get((tx, ty), TileType.WATER)
                if current in (TileType.GRASS, TileType.SAND, TileType.FOREST, TileType.CITY):
                    self.tile_type_map[(tx, ty)] = TileType.PATH

    # --- Drawing ---

    def _draw_terrain(self, ctx: RenderContext):
        # Draw terrain tiles - iterate through all tiles in bounds
        for tx in range(self.w_tiles):
            for ty in range(self.h_tiles):
                if (tx, ty) not in self.tile_type_map:
                    continue
                ttype = self.tile_type_map[(tx, ty)]
                # Convert tile to world coords, then to screen
                world_x = tx * self.ts
                world_y = ty * self.ts
                screen_pos = ctx.camera.world_to_screen((world_x, world_y))
                self.tileset.draw_tile(ttype, ctx.screen, 
                                       screen_pos[0], screen_pos[1], 
                                       ctx.camera.zoom)

    def _draw_resources(self, ctx: RenderContext):
        # Already drawn in terrain usually, but maybe highlight?
        pass

    def _draw_cities(self, ctx: RenderContext):
        # Draw city tiles as path for now, or highlight
        for city in self.cities:
            for (tx, ty) in city['tiles']:
                # Draw a path overlay or replace with path
                 rect = (tx * self.ts, ty * self.ts, self.ts, self.ts)
                 s = pygame.Surface((self.ts, self.ts))
                 s.set_alpha(100)
                 s.fill((100, 0, 0)) # Red tint for city
                 ctx.screen.blit(s, (tx * self.ts, ty * self.ts))
                 
            # Center
            cx, cy = city['pos']
            pygame.draw.circle(ctx.screen, (255, 0, 0), ctx.camera.world_to_screen((cx, cy)), 5)

    def _draw_graph(self, ctx: RenderContext):
         for i, neighbors in enumerate(self.adj):
             p1 = self.sites[i]
             for n in neighbors:
                 if i < n: # Draw once
                     p2 = self.sites[n]
                     pygame.draw.line(ctx.screen, (50, 50, 50), 
                                      ctx.camera.world_to_screen(p1), 
                                      ctx.camera.world_to_screen(p2), 1)

    def _draw_poisson_seeds(self, ctx: RenderContext):
        """Step 1: Draw Poisson seed points"""
        for i, p in enumerate(self.sites):
            pygame.draw.circle(ctx.screen, (255, 255, 255), ctx.camera.world_to_screen(p), 3)
    
    def _draw_centers(self, ctx: RenderContext):
        for i, p in enumerate(self.sites):
            col = (255, 255, 255)
            if self.region_resource[i] == 3: col = (255, 215, 0)
            elif self.region_resource[i] == 2: col = (100, 100, 100)
            elif self.region_resource[i] == 1: col = (0, 100, 0)
            
            pygame.draw.circle(ctx.screen, col, ctx.camera.world_to_screen(p), 2)
    
    def _draw_voronoi_regions(self, ctx: RenderContext):
        """Draw Voronoi regions colored by elevation/land status"""
        for (tx, ty), owner_idx in self.owner_map.items():
            if tx >= self.w_tiles or ty >= self.h_tiles:
                continue
            if not ((tx, ty) in self.is_land.shape and 0 <= tx < self.is_land.shape[0] and 0 <= ty < self.is_land.shape[1]):
                continue
            if self.is_land[(tx, ty)]:
                # Color by elevation (green shades)
                elev = clamp(self.elevation_map[(tx, ty)], 0, 1)
                color = (int(40*elev), int(200 + 55*elev), int(30*elev))
            else:
                # Water (blue shades)
                elev = clamp(self.elevation_map[(tx, ty)], -0.5, 0.5)
                color = (int(40 - elev*40), int(120 - elev*40), int(255 - elev*40))
            draw_tile(ctx, (tx, ty), self.ts, color)
    
    def _draw_rivers(self, ctx: RenderContext):
        """Draw river edges on the graph"""
        for u, v in self.river_edges:
            p1 = self.sites[u]
            p2 = self.sites[v]
            pygame.draw.line(ctx.screen, (50, 150, 255), 
                             ctx.camera.world_to_screen(p1), 
                             ctx.camera.world_to_screen(p2), 3)
    
    def _draw_resources_graph(self, ctx: RenderContext):
        """Draw resource locations on graph centers"""
        for i, p in enumerate(self.sites):
            col = None
            if self.region_resource[i] == 3:  # Gold
                col = (255, 215, 0)
                pygame.draw.circle(ctx.screen, col, ctx.camera.world_to_screen(p), 8)
            elif self.region_resource[i] == 2:  # Stone
                col = (150, 150, 150)
                pygame.draw.circle(ctx.screen, col, ctx.camera.world_to_screen(p), 6)
            elif self.region_resource[i] == 1:  # Wood
                col = (0, 150, 0)
                pygame.draw.circle(ctx.screen, col, ctx.camera.world_to_screen(p), 5)
            else:
                # No resource - small white dot
                pygame.draw.circle(ctx.screen, (200, 200, 200), ctx.camera.world_to_screen(p), 2)

