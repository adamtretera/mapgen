
from enum import IntEnum
import pygame
from maie.common import NeighborIndex

class TileType(IntEnum):
    GRASS = 0
    WATER = 1
    SAND = 2
    FOREST = 3
    MOUNTAIN = 4  # Stone source
    GOLD = 5      # Gold source
    CITY = 6      # City
    PATH = 7      # Road

TileColors = {
    TileType.GRASS: (40, 220, 30),
    TileType.WATER: (40, 120, 255),
    TileType.SAND: (255, 255, 40),
    TileType.FOREST: (10, 100, 10),
    TileType.MOUNTAIN: (120, 120, 120),
    TileType.GOLD: (255, 215, 0),
    TileType.CITY: (180, 60, 60),
    TileType.PATH: (139, 69, 19), # SaddleBrown
}

class Tile:
    def __init__(self, type: TileType, size=16):
        self.type = type
        self.surf = pygame.Surface((size, size))
        self.surf.fill(TileColors[self.type])
        self.size = size

    def draw(self, surf, x, y, scale=1.0):
        size = (self.size*scale, self.size*scale)
        surf.blit(pygame.transform.scale(self.surf, size), (x, y))

class AgeTileSet:
    def __init__(self, base_tiles):
        self.tiles: list[Tile] = base_tiles
        self._by_type = {t.type: t for t in base_tiles}

    def draw_tile(self, tile_id: TileType, surf, x, y, scale=1.0):
        if tile_id in self._by_type:
            self._by_type[tile_id].draw(surf, x, y, scale)

def get_tileset(size):
    return AgeTileSet([
        Tile(TileType.GRASS, size),
        Tile(TileType.WATER, size),
        Tile(TileType.SAND, size),
        Tile(TileType.FOREST, size),
        Tile(TileType.MOUNTAIN, size),
        Tile(TileType.GOLD, size),
        Tile(TileType.CITY, size),
        Tile(TileType.PATH, size),
    ])

