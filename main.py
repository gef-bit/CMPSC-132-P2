import pygame
import random
import numpy as np
import time
from perlin_noise import PerlinNoise

# Settings
GRID_SIZE = 50
CELL_SIZE = 14
SPREAD_CHANCE = 0.3
FPS = 4

# States
EMPTY = 0
TREE = 1
FIRE = 2
FIRE_AGE1 = 3
BURNED = 4

# Colors
COLORS = {
    EMPTY: (0, 0, 0),
    TREE: (34, 139, 34),
    FIRE: (255, 0, 0),
    FIRE_AGE1: (200, 30, 30),
    BURNED: (50, 50, 50),
}

class Grid:
    def __init__(self, size, spread_chance=SPREAD_CHANCE+(random.random()-0.5)/10, wind_direction=random.choice(['N', 'E', 'S', 'W'])):
        self.size = size
        self.spread_chance = spread_chance
        self.wind_direction = wind_direction
        self.grid = self.create_grid()
        self.terrain = self.create_terrain()

    def create_terrain(self):
        noise = PerlinNoise(octaves=6)
        terrain = np.zeros((self.size, self.size))

        for y in range(self.size):
            for x in range(self.size):
                nx = x / self.size
                ny = y / self.size
                elevation = noise([nx, ny])
                terrain[y][x] = elevation

        # Normalize 0-100
        terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min()) * 100
        return terrain.astype(int)

    def create_grid(self):
        grid = np.zeros((self.size, self.size), dtype=int)
        for y in range(self.size):
            for x in range(self.size):
                if random.random() < 0.9:
                    grid[y][x] = TREE
        mid = self.size // 2
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if 0 <= mid + dy < self.size and 0 <= mid + dx < self.size:
                    grid[mid + dy][mid + dx] = FIRE
        return grid

    def get_neighbors(self, y, x):
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.size and 0 <= nx < self.size:
                neighbors.append((ny, nx))
        return neighbors

    def fire_spreads(self, y, x, ny, nx):
        wind = {'N': (-1, 0), 'S': (1, 0), 'W': (0, -1), 'E': (0, 1)}.get(self.wind_direction, (0, 0))
        base_chance = self.spread_chance

        if (ny - y, nx - x) == wind: # Wind influence
            base_chance += 0.4

        elevation_current = self.terrain[y][x] # Terrain influence
        elevation_next = self.terrain[ny][nx]

        elevation_diff = elevation_next - elevation_current

        if elevation_diff > 0:
            base_chance += 0.15 # Fire spreads faster uphill
        elif elevation_diff < 0: 
            base_chance -= 0.15 # Fire spreads slower downhill

        base_chance = max(0.0, min(1.0, base_chance))  # Clamp between 0 and 1

        return random.random() < base_chance

    def update(self):
        new_grid = self.grid.copy()
        fire_cells = np.argwhere((self.grid == FIRE) | (self.grid == FIRE_AGE1))

        if fire_cells.size == 0:
            return

        for y, x in fire_cells:
            for ny, nx in self.get_neighbors(y, x):
                if self.grid[ny][nx] == TREE:
                    if self.fire_spreads(y, x, ny, nx):
                        new_grid[ny][nx] = FIRE

            # Aging the fire instead of immediately burning out
            if self.grid[y][x] == FIRE:
                new_grid[y][x] = FIRE_AGE1
            elif self.grid[y][x] == FIRE_AGE1:
                new_grid[y][x] = BURNED

        self.grid = new_grid

    def draw(self, screen):
        for y in range(self.size):
            for x in range(self.size):
                state = self.grid[y][x]

                if state == TREE:
                    elevation = self.terrain[y][x] # Shade tree color based on elevation
                    brightness = int(50 + (elevation / 100) * 205) # Normalize elevation (0-100) to brightness (50-255)
                    color = (0, brightness, 0)  # Green varying by brightness
                else:
                    color = COLORS[state]

                pygame.draw.rect(screen, color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def set_wind_direction(self, new_direction):
        self.wind_direction = new_direction


class FireSimulator:
    def __init__(self, grid_size, cell_size, fps):
        pygame.init()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode((grid_size * cell_size, grid_size * cell_size))
        pygame.display.set_caption("Wildfire Simulation")
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.wind_direction = random.choice(['N', 'S', 'E', 'W'])
        self.grid = Grid(grid_size, SPREAD_CHANCE, self.wind_direction)
        self.running = True

        self.last_wind_change = time.time()
        self.wind_change_interval = 4  # seconds between wind changes
        self.simulation_start_time = time.time()
        self.wind_change_count = 0

    def draw_legend(self):
        font = pygame.font.Font(None, 30)
        legend_font = pygame.font.Font(None, 22)
        legend_title = font.render("Legend", True, (255, 255, 255))
        self.screen.blit(legend_title, (10, 10))

        legend_items = [
            ("Tree", (34, 139, 34)),
            ("Fire", (255, 0, 0)),
            ("Burned", (50, 50, 50)),
            ("Incombustible Object", (0, 0, 0)),
        ]

        for i, (label, color) in enumerate(legend_items):
            rect = pygame.Rect(10, 40 + i * 30, 20, 20)
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, (0, 0, 0), rect, width=1)

            label_text = legend_font.render(label, True, (255, 255, 255))
            self.screen.blit(label_text, (40, 40 + i * 30))

    def draw_wind_arrow(self):
        center_x = self.grid_size * self.cell_size - 60
        center_y = 40
        size = 20

        points = [
            (center_x, center_y - size),  # Arrow Tip
            (center_x - size//2, center_y + size//2),  # Left
            (center_x + size//2, center_y + size//2),  # Right
        ]

        # Rotate points based on wind direction
        angle = {
            'N': 0,
            'E': 90,
            'S': 180,
            'W': 270
        }.get(self.wind_direction, 0)
        rotated_points = self.rotate_points(points, (center_x, center_y), angle)

        pygame.draw.polygon(self.screen, (255, 255, 0), rotated_points)

    def rotate_points(self, points, center, angle):
        """Rotate points around center by angle degrees"""
        angle_rad = np.radians(angle)
        cos_theta, sin_theta = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy = center
        rotated = []
        for x, y in points:
            dx, dy = x - cx, y - cy
            new_x = cx + dx * cos_theta - dy * sin_theta
            new_y = cy + dx * sin_theta + dy * cos_theta
            rotated.append((new_x, new_y))
        return rotated

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.fps)

            if not self.fire_is_active():
                self.running = False  # Fire is extinguished

        restart = self.show_summary()
        if restart:
            self.grid = Grid(self.grid_size)
            self.simulation_start_time = time.time()
            self.wind_change_count = 0
            self.running = True
            self.run()

        else:
            pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        self.grid.update()
        current_time = time.time()
        if current_time - self.last_wind_change > self.wind_change_interval:
            self.change_wind()

    def draw(self):
        self.screen.fill((0, 0, 0))
        self.grid.draw(self.screen)
        self.draw_legend()
        self.draw_wind_arrow()
        pygame.display.flip()

    def change_wind(self):
        self.wind_direction = random.choice(['N', 'S', 'E', 'W'])
        self.grid.set_wind_direction(self.wind_direction)
        self.last_wind_change = time.time()
        self.wind_change_count += 1

    def fire_is_active(self):
        return np.any(self.grid.grid == FIRE) | np.any(self.grid.grid == FIRE_AGE1)

    def show_summary(self):
        total_cells = self.grid_size * self.grid_size
        trees_remaining = np.sum(self.grid.grid == TREE)
        burned_cells = np.sum(self.grid.grid == BURNED)
        empty_cells = np.sum(self.grid.grid == EMPTY)

        if (trees_remaining + burned_cells) > 0:
            burn_percentage = (burned_cells / (trees_remaining + burned_cells)) * 100
        else:
            burn_percentage = 0

        elapsed_time = time.time() - self.simulation_start_time

        summary_lines = [
            " Simulation Summary ",
            f"Total cells: {total_cells}",
            f"Trees remaining: {trees_remaining}",
            f"Trees burned: {burned_cells}",
            f"Empty/incombustible cells: {empty_cells}",
            f"Percentage of trees burned: {burn_percentage:.2f}%",
            f"Time elapsed: {elapsed_time:.2f} seconds",
            f"Wind direction changes: {self.wind_change_count}",
        ]

        return self.draw_summary_popup(summary_lines)

    def draw_summary_popup(self, lines):
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 28)

        popup_width = self.grid_size * self.cell_size // 2
        popup_height = len(lines) * 40 + 100
        popup_x = (self.grid_size * self.cell_size - popup_width) // 2
        popup_y = (self.grid_size * self.cell_size - popup_height) // 2

        popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)

        button_width = 160
        button_height = 40
        button_margin = 20

        restart_button = pygame.Rect(
            popup_rect.centerx - button_width - button_margin//2,
            popup_rect.bottom - button_height - 20,
            button_width,
            button_height
        )
        exit_button = pygame.Rect(
            popup_rect.centerx + button_margin//2,
            popup_rect.bottom - button_height - 20,
            button_width,
            button_height
        )

        waiting = True
        restart = False

        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
                    restart = False
                if event.type == pygame.KEYDOWN:
                    waiting = False
                    restart = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if restart_button.collidepoint(event.pos):
                        restart = True
                        waiting = False
                    if exit_button.collidepoint(event.pos):
                        restart = False
                        waiting = False

            self.screen.fill((0, 0, 0))
            self.grid.draw(self.screen)
            self.draw_wind_arrow()
            self.draw_legend()

            # Draw popup background
            pygame.draw.rect(self.screen, (30, 30, 30), popup_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), popup_rect, 2)

            # Draw summary lines
            for i, line in enumerate(lines):
                if i == 0:
                    text_surface = font.render(line, True, (255, 100, 100))
                else:
                    text_surface = small_font.render(line, True, (255, 255, 255))
                text_rect = text_surface.get_rect(center=(popup_rect.centerx, popup_rect.top + 30 + i * 40))
                self.screen.blit(text_surface, text_rect)

            pygame.draw.rect(self.screen, (70, 130, 180), restart_button)  # Blue button
            pygame.draw.rect(self.screen, (200, 50, 50), exit_button)      # Red button

            restart_text = small_font.render("Restart", True, (255, 255, 255))
            exit_text = small_font.render("Exit", True, (255, 255, 255))

            self.screen.blit(restart_text, restart_text.get_rect(center=restart_button.center))
            self.screen.blit(exit_text, exit_text.get_rect(center=exit_button.center))

            pygame.display.flip()
            self.clock.tick(30)

        return restart

    def restart_simulation(self):
        self.grid = Grid(self.grid_size)
        self.running = True
        self.run()


if __name__ == "__main__":
    sim = FireSimulator(GRID_SIZE, CELL_SIZE, FPS)
    sim.run()
